/*
  Random property tests for Handmade Math matrices, projections, and
  quaternions.

  Every property runs 500,000 independently generated inputs by default.  The
  inputs are split evenly across five classes:

    regular, near_singular, tiny, zero_vector, huge

  Each input is evaluated twice and the complete observations are compared
  byte-for-byte.  If a property fails, the input is automatically reduced with
  a deterministic greedy shrinker until no single scalar can be made simpler
  while preserving the failure.

  Build/run from test/:

      make property

  Optional:

      ./build/property_test --count 10000
      ./build/property_test --seed 1234
*/

#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../HandmadeMath.h"

#ifndef PROPERTY_DEFAULT_COUNT
#define PROPERTY_DEFAULT_COUNT 500000
#endif

#define PROPERTY_MAX_FLOATS 64
#define PROPERTY_MAX_INTS 8
#define PROPERTY_MAX_OBSERVATIONS 96
#define PROPERTY_CLASS_COUNT 5

typedef enum PropertyClass
{
    CLASS_REGULAR = 0,
    CLASS_NEAR_SINGULAR = 1,
    CLASS_TINY = 2,
    CLASS_ZERO_VECTOR = 3,
    CLASS_HUGE = 4,
} PropertyClass;

static const char *CLASS_NAMES[PROPERTY_CLASS_COUNT] = {
    "regular",
    "near_singular",
    "tiny",
    "zero_vector",
    "huge",
};

typedef struct TestCase
{
    int Class;
    int NumFloats;
    int NumInts;
    float Floats[PROPERTY_MAX_FLOATS];
    int Ints[PROPERTY_MAX_INTS];
} TestCase;

typedef struct Observation
{
    int NumFloats;
    int Failure;
    float Floats[PROPERTY_MAX_OBSERVATIONS];
} Observation;

typedef struct Property
{
    const char *Name;
    int (*Evaluate)(const TestCase *Case, Observation *Obs);
    void (*PrintCase)(const TestCase *Case);
} Property;

typedef struct Rng
{
    uint64_t State;
} Rng;

enum FailureCode
{
    FAILURE_NONE = 0,
    FAILURE_NONFINITE,
    FAILURE_ORTHOGONALITY,
    FAILURE_INVERSE_ROUNDTRIP,
    FAILURE_INTERPOLATION_ENDPOINT,
    FAILURE_INTERPOLATION_UNIT_LENGTH,
    FAILURE_PROJECTION_RANGE,
    FAILURE_PROJECTION_HOMOGENEOUS,
    FAILURE_QUAT_MATRIX_ROTATION,
    FAILURE_QUAT_MATRIX_DIRECT_ROTATION,
    FAILURE_QUAT_MATRIX_ROUNDTRIP,
    FAILURE_DETERMINISM,
};

static const char *FailureName(int Failure)
{
    switch (Failure)
    {
        case FAILURE_NONE: return "none";
        case FAILURE_NONFINITE: return "non-finite result";
        case FAILURE_ORTHOGONALITY: return "matrix is not orthogonal";
        case FAILURE_INVERSE_ROUNDTRIP: return "inverse round-trip is not identity";
        case FAILURE_INTERPOLATION_ENDPOINT: return "interpolation endpoint mismatch";
        case FAILURE_INTERPOLATION_UNIT_LENGTH: return "interpolation result is not unit length";
        case FAILURE_PROJECTION_RANGE: return "projected point is outside clip/NDC range";
        case FAILURE_PROJECTION_HOMOGENEOUS: return "projection has invalid homogeneous W";
        case FAILURE_QUAT_MATRIX_ROTATION: return "quaternion and matrix disagree on vector rotation";
        case FAILURE_QUAT_MATRIX_DIRECT_ROTATION: return "QToM4 and direct axis-angle rotation disagree";
        case FAILURE_QUAT_MATRIX_ROUNDTRIP: return "M4ToQ(QToM4(q)) disagrees with q";
        case FAILURE_DETERMINISM: return "the same input produced different bits";
        default: return "unknown failure";
    }
}

static uint64_t HashFloat(uint64_t Hash, float Value)
{
    uint32_t Bits;
    memcpy(&Bits, &Value, sizeof(Bits));
    Hash ^= Bits;
    Hash *= UINT64_C(1099511628211);
    return Hash;
}

static uint64_t HashInt(uint64_t Hash, int Value)
{
    Hash ^= (uint32_t)Value;
    Hash *= UINT64_C(1099511628211);
    return Hash;
}

static void RngInit(Rng *R, uint64_t Seed)
{
    R->State = Seed ? Seed : UINT64_C(0x9e3779b97f4a7c15);
}

static uint64_t RngNextU64(Rng *R)
{
    uint64_t X = R->State;
    X ^= X >> 12;
    X ^= X << 25;
    X ^= X >> 27;
    R->State = X;
    return X * UINT64_C(2685821657736338717);
}

static uint32_t RngNextU32(Rng *R)
{
    return (uint32_t)(RngNextU64(R) >> 32);
}

static float RngUnit(Rng *R)
{
    return (float)(RngNextU32(R) >> 8) * (1.0f / 16777216.0f);
}

static float RngSigned(Rng *R)
{
    return RngUnit(R) * 2.0f - 1.0f;
}

static int RngRangeInt(Rng *R, int Min, int MaxInclusive)
{
    return Min + (int)(RngNextU32(R) % (uint32_t)(MaxInclusive - Min + 1));
}

static float RngRange(Rng *R, float Min, float Max)
{
    return Min + RngUnit(R) * (Max - Min);
}

static float AbsF(float X)
{
    return X < 0.0f ? -X : X;
}

static float MaxF(float A, float B)
{
    return A > B ? A : B;
}

static int NearlyEqual(float A, float B, float RelativeTolerance, float AbsoluteTolerance)
{
    float Scale;
    float Difference;

    if (A == B)
    {
        return 1;
    }
    if (!isfinite(A) || !isfinite(B))
    {
        return 0;
    }

    Scale = MaxF(1.0f, MaxF(AbsF(A), AbsF(B)));
    Difference = AbsF(A - B);
    return Difference <= AbsoluteTolerance + RelativeTolerance * Scale;
}

static int NearlyEqualScaled(float A, float B, float RelativeTolerance,
                             float AbsoluteTolerance, float Scale)
{
    float Difference;

    if (A == B)
    {
        return 1;
    }
    if (!isfinite(A) || !isfinite(B))
    {
        return 0;
    }

    Difference = AbsF(A - B);
    return Difference <= AbsoluteTolerance +
        RelativeTolerance * MaxF(1.0f, MaxF(Scale, MaxF(AbsF(A), AbsF(B))));
}

static float ObservationMax(float Current, float Candidate)
{
    return Candidate > Current ? Candidate : Current;
}

static void ObserveFloat(Observation *Obs, float Value)
{
    if (Obs->NumFloats < PROPERTY_MAX_OBSERVATIONS)
    {
        Obs->Floats[Obs->NumFloats++] = Value;
    }
    if (!isfinite(Value))
    {
        Obs->Failure = FAILURE_NONFINITE;
    }
}

static void ObserveQuat(Observation *Obs, HMM_Quat Q)
{
    ObserveFloat(Obs, Q.X);
    ObserveFloat(Obs, Q.Y);
    ObserveFloat(Obs, Q.Z);
    ObserveFloat(Obs, Q.W);
}

static HMM_Vec3 NormalizeV3Double(HMM_Vec3 V)
{
    double X = V.X;
    double Y = V.Y;
    double Z = V.Z;
    double Length = sqrt(X * X + Y * Y + Z * Z);

    if (Length == 0.0)
    {
        return HMM_V3(0.0f, 1.0f, 0.0f);
    }

    V.X = (float)(X / Length);
    V.Y = (float)(Y / Length);
    V.Z = (float)(Z / Length);
    return V;
}

static HMM_Quat NormalizeQDouble(HMM_Quat Q)
{
    double X = Q.X;
    double Y = Q.Y;
    double Z = Q.Z;
    double W = Q.W;
    double Length = sqrt(X * X + Y * Y + Z * Z + W * W);

    if (Length == 0.0)
    {
        return HMM_Q(0.0f, 0.0f, 0.0f, 1.0f);
    }

    Q.X = (float)(X / Length);
    Q.Y = (float)(Y / Length);
    Q.Z = (float)(Z / Length);
    Q.W = (float)(W / Length);
    return Q;
}

static int QuatEquivalent(HMM_Quat A, HMM_Quat B, float Tolerance)
{
    float Same = 1.0f;
    float Opposite = 1.0f;

    Same = Same && NearlyEqual(A.X, B.X, 0.0f, Tolerance);
    Same = Same && NearlyEqual(A.Y, B.Y, 0.0f, Tolerance);
    Same = Same && NearlyEqual(A.Z, B.Z, 0.0f, Tolerance);
    Same = Same && NearlyEqual(A.W, B.W, 0.0f, Tolerance);

    Opposite = Opposite && NearlyEqual(A.X, -B.X, 0.0f, Tolerance);
    Opposite = Opposite && NearlyEqual(A.Y, -B.Y, 0.0f, Tolerance);
    Opposite = Opposite && NearlyEqual(A.Z, -B.Z, 0.0f, Tolerance);
    Opposite = Opposite && NearlyEqual(A.W, -B.W, 0.0f, Tolerance);

    return Same || Opposite;
}

static HMM_Quat RandomQuat(Rng *Rng, float Scale)
{
    return HMM_Q(
        RngSigned(Rng) * Scale,
        RngSigned(Rng) * Scale,
        RngSigned(Rng) * Scale,
        RngSigned(Rng) * Scale);
}

static HMM_Quat QuatFromRandomAxisAngle(Rng *Rng, float Angle)
{
    HMM_Vec3 Axis = HMM_V3(RngSigned(Rng), RngSigned(Rng), RngSigned(Rng));
    Axis = NormalizeV3Double(Axis);

    return HMM_Q(
        Axis.X * sinf(Angle * 0.5f),
        Axis.Y * sinf(Angle * 0.5f),
        Axis.Z * sinf(Angle * 0.5f),
        cosf(Angle * 0.5f));
}

static HMM_Mat3 TopLeftM3(HMM_Mat4 M)
{
    HMM_Mat3 Result;
    int Column;
    int Row;

    for (Column = 0; Column < 3; ++Column)
    {
        for (Row = 0; Row < 3; ++Row)
        {
            Result.Elements[Column][Row] = M.Elements[Column][Row];
        }
    }
    return Result;
}

/*
 * Property 1: orthogonality
 *
 * QToM4 is used to construct 2x2, 3x3, and 4x4 rotation matrices from the same
 * voxelized random quaternion.  The library's transpose and multiply functions
 * must produce an identity matrix.
 */

static int EvaluateOrthogonality(const TestCase *Case, Observation *Obs)
{
    HMM_Quat Q = HMM_Q(
        Case->Floats[0],
        Case->Floats[1],
        Case->Floats[2],
        Case->Floats[3]);
    HMM_Mat4 M = HMM_QToM4(Q);
    HMM_Mat2 M2;
    HMM_Mat3 M3 = TopLeftM3(M);
    HMM_Mat2 P2 = HMM_MulM2(HMM_TransposeM2(M2), M2);
    HMM_Mat3 P3 = HMM_MulM3(HMM_TransposeM3(M3), M3);
    HMM_Mat4 P4 = HMM_MulM4(HMM_TransposeM4(M), M);
    float MaxError = 0.0f;
    int Column;
    int Row;
    int Dimension = Case->Ints[0];

    if (Dimension == 2)
    {
        HMM_Quat QN = NormalizeQDouble(Q);
        float Angle = 2.0f * atan2f(QN.Z, QN.W);
        float Cosine = cosf(Angle);
        float Sine = sinf(Angle);
        M2.Elements[0][0] = Cosine;
        M2.Elements[0][1] = Sine;
        M2.Elements[1][0] = -Sine;
        M2.Elements[1][1] = Cosine;
        P2 = HMM_MulM2(HMM_TransposeM2(M2), M2);
    }

    for (Column = 0; Column < Dimension; ++Column)
    {
        for (Row = 0; Row < Dimension; ++Row)
        {
            float Expected = Column == Row ? 1.0f : 0.0f;
            float Actual = Dimension == 2
                ? P2.Elements[Column][Row]
                : (Dimension == 3 ? P3.Elements[Column][Row] : P4.Elements[Column][Row]);

            ObserveFloat(Obs, Actual);
            MaxError = ObservationMax(MaxError, AbsF(Actual - Expected));
        }
    }
    ObserveFloat(Obs, MaxError);

    if (Obs->Failure == FAILURE_NONFINITE)
    {
        return 1;
    }

    {
        float Tolerance = Case->Class == CLASS_NEAR_SINGULAR ? 5.0e-4f : 2.0e-4f;
        if (MaxError > Tolerance)
        {
            Obs->Failure = FAILURE_ORTHOGONALITY;
        }
    }
    return 1;
}

static void PrintOrthogonalityCase(const TestCase *Case)
{
    printf("dimension=%d q=(%a, %a, %a, %a) [%.9g, %.9g, %.9g, %.9g]",
           Case->Ints[0],
           Case->Floats[0], Case->Floats[1], Case->Floats[2], Case->Floats[3],
           Case->Floats[0], Case->Floats[1], Case->Floats[2], Case->Floats[3]);
}

static void GenerateOrthogonalityCase(TestCase *Case, Rng *Rng, int Class)
{
    HMM_Quat Q;

    memset(Case, 0, sizeof(*Case));
    Case->Class = Class;
    Case->NumFloats = 4;
    Case->NumInts = 1;
    Case->Ints[0] = RngRangeInt(Rng, 2, 4);

    switch (Class)
    {
        case CLASS_REGULAR:
        {
            Q = RandomQuat(Rng, 1.0f);
            break;
        }
        case CLASS_NEAR_SINGULAR:
        {
            float Angle = RngRange(Rng, 1.0e-6f, 1.0e-4f);
            Q = QuatFromRandomAxisAngle(Rng, Angle);
            break;
        }
        case CLASS_TINY:
        {
            Q = RandomQuat(Rng, 1.0e-18f);
            break;
        }
        case CLASS_ZERO_VECTOR:
        {
            Q = HMM_Q(0.0f, 0.0f, 0.0f, RngSigned(Rng) < 0.0f ? -1.0f : 1.0f);
            break;
        }
        case CLASS_HUGE:
        default:
        {
            Q = RandomQuat(Rng, 1.0e18f);
            break;
        }
    }

    Case->Floats[0] = Q.X;
    Case->Floats[1] = Q.Y;
    Case->Floats[2] = Q.Z;
    Case->Floats[3] = Q.W;
}

/*
 * Property 2: inverse round-trip
 *
 * M = R * diag(s).  R is an orthogonal rotation matrix and the singular values
 * exercise regular, near-singular, tiny, zero-axis, and huge cases.  Both
 * M * inverse(M) and inverse(M) * M must be identity.
 */

static void BuildScaledRotation(const TestCase *Case, float Matrix[4][4])
{
    HMM_Quat Q = HMM_Q(
        Case->Floats[0],
        Case->Floats[1],
        Case->Floats[2],
        Case->Floats[3]);
    HMM_Mat4 R4 = HMM_QToM4(Q);
    int Dimension = Case->Ints[0];
    int Column;
    int Row;

    if (Dimension == 2)
    {
        HMM_Quat QN = NormalizeQDouble(Q);
        float Angle = 2.0f * atan2f(QN.Z, QN.W);
        float Cosine = cosf(Angle);
        float Sine = sinf(Angle);
        Matrix[0][0] = Cosine;
        Matrix[0][1] = Sine;
        Matrix[1][0] = -Sine;
        Matrix[1][1] = Cosine;
        for (Column = 0; Column < 2; ++Column)
        {
            for (Row = 0; Row < 2; ++Row)
            {
                Matrix[Column][Row] *= Case->Floats[4 + Column];
            }
        }
        return;
    }

    for (Column = 0; Column < Dimension; ++Column)
    {
        float Scale = Case->Floats[4 + Column];
        for (Row = 0; Row < Dimension; ++Row)
        {
            Matrix[Column][Row] = R4.Elements[Column][Row] * Scale;
        }
    }
}

static int EvaluateInverse(const TestCase *Case, Observation *Obs)
{
    float Matrix[4][4];
    float MaxError = 0.0f;
    float Tolerance;
    int Dimension = Case->Ints[0];
    int Column;
    int Row;

    memset(Matrix, 0, sizeof(Matrix));
    BuildScaledRotation(Case, Matrix);

    {
        double Determinant = 1.0;
        int Valid = 1;

        for (Column = 0; Column < Dimension; ++Column)
        {
            for (Row = 0; Row < Dimension; ++Row)
            {
                if (!isfinite(Matrix[Column][Row]))
                {
                    Valid = 0;
                }
            }
            Determinant *= (double)Case->Floats[4 + Column];
        }
        if (!Valid || Determinant == 0.0 || !isfinite(Determinant))
        {
            return 0;
        }
    }

    if (Dimension == 2)
    {
        HMM_Mat2 M;
        HMM_Mat2 Inv;
        HMM_Mat2 Left;
        HMM_Mat2 Right;

        M.Elements[0][0] = Matrix[0][0];
        M.Elements[0][1] = Matrix[0][1];
        M.Elements[1][0] = Matrix[1][0];
        M.Elements[1][1] = Matrix[1][1];
        Inv = HMM_InvGeneralM2(M);
        Left = HMM_MulM2(M, Inv);
        Right = HMM_MulM2(Inv, M);

        for (Column = 0; Column < 2; ++Column)
        {
            for (Row = 0; Row < 2; ++Row)
            {
                float Expected = Column == Row ? 1.0f : 0.0f;
                ObserveFloat(Obs, Left.Elements[Column][Row]);
                ObserveFloat(Obs, Right.Elements[Column][Row]);
                MaxError = ObservationMax(MaxError, AbsF(Left.Elements[Column][Row] - Expected));
                MaxError = ObservationMax(MaxError, AbsF(Right.Elements[Column][Row] - Expected));
            }
        }
    }
    else if (Dimension == 3)
    {
        HMM_Mat3 M;
        HMM_Mat3 Inv;
        HMM_Mat3 Left;
        HMM_Mat3 Right;

        for (Column = 0; Column < 3; ++Column)
        {
            for (Row = 0; Row < 3; ++Row)
            {
                M.Elements[Column][Row] = Matrix[Column][Row];
            }
        }
        Inv = HMM_InvGeneralM3(M);
        Left = HMM_MulM3(M, Inv);
        Right = HMM_MulM3(Inv, M);

        for (Column = 0; Column < 3; ++Column)
        {
            for (Row = 0; Row < 3; ++Row)
            {
                float Expected = Column == Row ? 1.0f : 0.0f;
                ObserveFloat(Obs, Left.Elements[Column][Row]);
                ObserveFloat(Obs, Right.Elements[Column][Row]);
                MaxError = ObservationMax(MaxError, AbsF(Left.Elements[Column][Row] - Expected));
                MaxError = ObservationMax(MaxError, AbsF(Right.Elements[Column][Row] - Expected));
            }
        }
    }
    else
    {
        HMM_Mat4 M;
        HMM_Mat4 Inv;
        HMM_Mat4 Left;
        HMM_Mat4 Right;

        for (Column = 0; Column < 4; ++Column)
        {
            for (Row = 0; Row < 4; ++Row)
            {
                M.Elements[Column][Row] = Matrix[Column][Row];
            }
        }
        Inv = HMM_InvGeneralM4(M);
        Left = HMM_MulM4(M, Inv);
        Right = HMM_MulM4(Inv, M);

        for (Column = 0; Column < 4; ++Column)
        {
            for (Row = 0; Row < 4; ++Row)
            {
                float Expected = Column == Row ? 1.0f : 0.0f;
                ObserveFloat(Obs, Left.Elements[Column][Row]);
                ObserveFloat(Obs, Right.Elements[Column][Row]);
                MaxError = ObservationMax(MaxError, AbsF(Left.Elements[Column][Row] - Expected));
                MaxError = ObservationMax(MaxError, AbsF(Right.Elements[Column][Row] - Expected));
            }
        }
    }

    ObserveFloat(Obs, MaxError);
    if (Obs->Failure == FAILURE_NONFINITE)
    {
        return 1;
    }

    Tolerance = Case->Class == CLASS_NEAR_SINGULAR ? 5.0e-2f : 5.0e-4f;
    if (MaxError > Tolerance)
    {
        Obs->Failure = FAILURE_INVERSE_ROUNDTRIP;
    }
    return 1;
}

static void PrintInverseCase(const TestCase *Case)
{
    int Index;
    printf("dimension=%d singular_values=(", Case->Ints[0]);
    for (Index = 0; Index < Case->Ints[0]; ++Index)
    {
        printf("%s%a", Index == 0 ? "" : ", ", Case->Floats[4 + Index]);
    }
    printf(") q=(%a, %a, %a, %a)",
           Case->Floats[0], Case->Floats[1], Case->Floats[2], Case->Floats[3]);
}

static void GenerateInverseCase(TestCase *Case, Rng *Rng, int Class)
{
    HMM_Quat Q;
    int Dimension;
    int Index;

    memset(Case, 0, sizeof(*Case));
    Case->Class = Class;
    Case->NumFloats = 8;
    Case->NumInts = 1;
    Dimension = RngRangeInt(Rng, 2, 4);
    Case->Ints[0] = Dimension;

    switch (Class)
    {
        case CLASS_REGULAR:
        {
            Q = RandomQuat(Rng, 1.0f);
            for (Index = 0; Index < Dimension; ++Index)
            {
                Case->Floats[4 + Index] = RngRange(Rng, 0.25f, 1.75f);
            }
            break;
        }
        case CLASS_NEAR_SINGULAR:
        {
            Q = RandomQuat(Rng, 1.0f);
            Case->Floats[4] = RngRange(Rng, 1.0e-4f, 2.0e-4f);
            for (Index = 1; Index < Dimension; ++Index)
            {
                Case->Floats[4 + Index] = RngRange(Rng, 0.5f, 2.0f);
            }
            break;
        }
        case CLASS_TINY:
        {
            float TinyScale = Dimension == 2 ? 1.0e-18f : (Dimension == 3 ? 1.0e-12f : 1.0e-9f);
            Q = RandomQuat(Rng, 1.0f);
            for (Index = 0; Index < Dimension; ++Index)
            {
                Case->Floats[4 + Index] = TinyScale;
            }
            break;
        }
        case CLASS_ZERO_VECTOR:
        {
            Q = HMM_Q(0.0f, 0.0f, 0.0f, 1.0f);
            for (Index = 0; Index < Dimension; ++Index)
            {
                Case->Floats[4 + Index] = RngRange(Rng, 0.25f, 1.75f);
            }
            break;
        }
        case CLASS_HUGE:
        default:
        {
            float HugeScale = Dimension == 2 ? 1.0e18f : (Dimension == 3 ? 1.0e12f : 1.0e9f);
            Q = RandomQuat(Rng, 1.0f);
            for (Index = 0; Index < Dimension; ++Index)
            {
                Case->Floats[4 + Index] = HugeScale;
            }
            break;
        }
    }

    Case->Floats[0] = Q.X;
    Case->Floats[1] = Q.Y;
    Case->Floats[2] = Q.Z;
    Case->Floats[3] = Q.W;
}

/*
 * Property 3: interpolation endpoints
 *
 * NLerp and SLerp must reproduce (up to quaternion sign) the normalized left
 * endpoint at t=0 and the normalized right endpoint at t=1.
 */

static int EvaluateInterpolation(const TestCase *Case, Observation *Obs)
{
    HMM_Quat Q0 = HMM_Q(Case->Floats[0], Case->Floats[1], Case->Floats[2], Case->Floats[3]);
    HMM_Quat Q1 = HMM_Q(Case->Floats[4], Case->Floats[5], Case->Floats[6], Case->Floats[7]);
    HMM_Quat Expected0 = NormalizeQDouble(Q0);
    HMM_Quat Expected1 = NormalizeQDouble(Q1);
    HMM_Quat Results[4];
    int Index;
    float Tolerance = Case->Class == CLASS_NEAR_SINGULAR ? 1.0e-3f : 2.0e-4f;

    Results[0] = HMM_NLerp(Q0, 0.0f, Q1);
    Results[1] = HMM_NLerp(Q0, 1.0f, Q1);
    Results[2] = HMM_SLerp(Q0, 0.0f, Q1);
    Results[3] = HMM_SLerp(Q0, 1.0f, Q1);

    for (Index = 0; Index < 4; ++Index)
    {
        HMM_Quat Expected = (Index == 0 || Index == 2) ? Expected0 : Expected1;
        float Length = HMM_LenV4(HMM_V4(Results[Index].X, Results[Index].Y, Results[Index].Z, Results[Index].W));

        ObserveQuat(Obs, Results[Index]);
        ObserveFloat(Obs, Length);

        if (!QuatEquivalent(Results[Index], Expected, Tolerance))
        {
            Obs->Failure = FAILURE_INTERPOLATION_ENDPOINT;
        }
        if (!NearlyEqual(Length, 1.0f, 2.0e-4f, 2.0e-4f))
        {
            Obs->Failure = FAILURE_INTERPOLATION_UNIT_LENGTH;
        }
    }
    return 1;
}

static void PrintInterpolationCase(const TestCase *Case)
{
    printf("q0=(%a, %a, %a, %a) q1=(%a, %a, %a, %a)",
           Case->Floats[0], Case->Floats[1], Case->Floats[2], Case->Floats[3],
           Case->Floats[4], Case->Floats[5], Case->Floats[6], Case->Floats[7]);
}

static void GenerateInterpolationCase(TestCase *Case, Rng *Rng, int Class)
{
    HMM_Quat Q0;
    HMM_Quat Q1;

    memset(Case, 0, sizeof(*Case));
    Case->Class = Class;
    Case->NumFloats = 8;

    switch (Class)
    {
        case CLASS_REGULAR:
        {
            Q0 = RandomQuat(Rng, 1.0f);
            Q1 = RandomQuat(Rng, 1.0f);
            break;
        }
        case CLASS_NEAR_SINGULAR:
        {
            Q0 = RandomQuat(Rng, 1.0f);
            Q1 = HMM_Q(
                -Q0.X + RngSigned(Rng) * 1.0e-7f,
                -Q0.Y + RngSigned(Rng) * 1.0e-7f,
                -Q0.Z + RngSigned(Rng) * 1.0e-7f,
                -Q0.W + RngSigned(Rng) * 1.0e-7f);
            break;
        }
        case CLASS_TINY:
        {
            Q0 = RandomQuat(Rng, 1.0e-18f);
            Q1 = RandomQuat(Rng, 1.0e-18f);
            break;
        }
        case CLASS_ZERO_VECTOR:
        {
            Q0 = HMM_Q(0.0f, 0.0f, 0.0f, 1.0f);
            Q1 = HMM_Q(0.0f, 0.0f, 0.0f, -1.0f);
            break;
        }
        case CLASS_HUGE:
        default:
        {
            Q0 = RandomQuat(Rng, 1.0e18f);
            Q1 = RandomQuat(Rng, 1.0e18f);
            break;
        }
    }

    Case->Floats[0] = Q0.X;
    Case->Floats[1] = Q0.Y;
    Case->Floats[2] = Q0.Z;
    Case->Floats[3] = Q0.W;
    Case->Floats[4] = Q1.X;
    Case->Floats[5] = Q1.Y;
    Case->Floats[6] = Q1.Z;
    Case->Floats[7] = Q1.W;
}

/*
 * Property 4: projection clip-space range
 *
 * Orthographic and perspective projections are tested in all four
 * handedness/NDC combinations.  The eight frustum corners must map to the
 * expected X/Y/Z NDC boundary, and center points exercise zero X/Y vectors.
 */

static void BuildProjection(const TestCase *Case, HMM_Mat4 *Projection)
{
    int Perspective = Case->Ints[0];
    int LeftHanded = Case->Ints[1];
    int ZeroToOne = Case->Ints[2];

    if (Perspective)
    {
        float Fov = Case->Floats[0];
        float Aspect = Case->Floats[1];
        float Near = Case->Floats[4];
        float Far = Case->Floats[5];

        if (!LeftHanded && !ZeroToOne) *Projection = HMM_Perspective_RH_NO(Fov, Aspect, Near, Far);
        else if (!LeftHanded && ZeroToOne) *Projection = HMM_Perspective_RH_ZO(Fov, Aspect, Near, Far);
        else if (LeftHanded && !ZeroToOne) *Projection = HMM_Perspective_LH_NO(Fov, Aspect, Near, Far);
        else *Projection = HMM_Perspective_LH_ZO(Fov, Aspect, Near, Far);
    }
    else
    {
        float Left = Case->Floats[0];
        float Right = Case->Floats[1];
        float Bottom = Case->Floats[2];
        float Top = Case->Floats[3];
        float Near = Case->Floats[4];
        float Far = Case->Floats[5];

        if (!LeftHanded && !ZeroToOne) *Projection = HMM_Orthographic_RH_NO(Left, Right, Bottom, Top, Near, Far);
        else if (!LeftHanded && ZeroToOne) *Projection = HMM_Orthographic_RH_ZO(Left, Right, Bottom, Top, Near, Far);
        else if (LeftHanded && !ZeroToOne) *Projection = HMM_Orthographic_LH_NO(Left, Right, Bottom, Top, Near, Far);
        else *Projection = HMM_Orthographic_LH_ZO(Left, Right, Bottom, Top, Near, Far);
    }
}

static int EvaluateProjection(const TestCase *Case, Observation *Obs)
{
    HMM_Mat4 Projection;
    int Perspective = Case->Ints[0];
    int LeftHanded = Case->Ints[1];
    int ZeroToOne = Case->Ints[2];
    float ZMin = ZeroToOne ? 0.0f : -1.0f;
    float ZMax = 1.0f;
    float Tolerance = Case->Class == CLASS_NEAR_SINGULAR ? 5.0e-3f : 2.0e-4f;
    float Near = Case->Floats[4];
    float Far = Case->Floats[5];
    float ZNear = LeftHanded ? Near : -Near;
    float ZFar = LeftHanded ? Far : -Far;
    int PointCount = Case->Class == CLASS_ZERO_VECTOR ? 2 : 8;
    int PointIndex;

    if (Perspective)
    {
        if (!(Case->Floats[0] > 0.0f && Case->Floats[0] < HMM_PI32) ||
            !(Case->Floats[1] > 0.0f) ||
            !(Case->Floats[4] > 0.0f && Case->Floats[5] > Case->Floats[4]))
        {
            return 0;
        }
    }
    else
    {
        if (!(Case->Floats[1] > Case->Floats[0]) ||
            !(Case->Floats[3] > Case->Floats[2]) ||
            !(Case->Floats[4] > 0.0f && Case->Floats[5] > Case->Floats[4]))
        {
            return 0;
        }
    }

    BuildProjection(Case, &Projection);
    ObserveFloat(Obs, Projection.Elements[0][0]);
    ObserveFloat(Obs, Projection.Elements[1][1]);
    ObserveFloat(Obs, Projection.Elements[2][2]);
    ObserveFloat(Obs, Projection.Elements[2][3]);
    ObserveFloat(Obs, Projection.Elements[3][2]);

    for (PointIndex = 0; PointIndex < PointCount; ++PointIndex)
    {
        int DepthIndex = PointIndex >= 4 ? 1 : 0;
        int XSign = (PointIndex & 1) ? 1 : -1;
        int YSign = (PointIndex & 2) ? 1 : -1;
        float Z = DepthIndex ? ZFar : ZNear;
        float X = 0.0f;
        float Y = 0.0f;
        float ExpectedX = 0.0f;
        float ExpectedY = 0.0f;
        float ExpectedZ = DepthIndex ? ZMax : ZMin;
        HMM_Vec4 Point;
        HMM_Vec4 Clip;

        if (Perspective)
        {
            if (Case->Class != CLASS_ZERO_VECTOR)
            {
                float HalfHeight = tanf(Case->Floats[0] * 0.5f) * AbsF(Z);
                float HalfWidth = HalfHeight * Case->Floats[1];
                X = (float)XSign * HalfWidth;
                Y = (float)YSign * HalfHeight;
                ExpectedX = (float)XSign;
                ExpectedY = (float)YSign;
            }
        }
        else if (Case->Class != CLASS_ZERO_VECTOR)
        {
            float Left = Case->Floats[0];
            float Right = Case->Floats[1];
            float Bottom = Case->Floats[2];
            float Top = Case->Floats[3];
            X = XSign < 0 ? Left : Right;
            Y = YSign < 0 ? Bottom : Top;
            ExpectedX = (float)XSign;
            ExpectedY = (float)YSign;
        }

        Point = HMM_V4(X, Y, Z, 1.0f);
        Clip = HMM_MulM4V4(Projection, Point);
        ObserveFloat(Obs, Clip.X);
        ObserveFloat(Obs, Clip.Y);
        ObserveFloat(Obs, Clip.Z);
        ObserveFloat(Obs, Clip.W);

        if (!isfinite(Clip.W) || Clip.W == 0.0f ||
            (Perspective && Clip.W <= 0.0f))
        {
            Obs->Failure = FAILURE_PROJECTION_HOMOGENEOUS;
            continue;
        }

        {
            float NdcX = Clip.X / Clip.W;
            float NdcY = Clip.Y / Clip.W;
            float NdcZ = Clip.Z / Clip.W;
            float RangeEpsilon = Tolerance;

            ObserveFloat(Obs, NdcX);
            ObserveFloat(Obs, NdcY);
            ObserveFloat(Obs, NdcZ);

            if (!NearlyEqual(NdcX, ExpectedX, Tolerance, Tolerance) ||
                !NearlyEqual(NdcY, ExpectedY, Tolerance, Tolerance) ||
                !NearlyEqual(NdcZ, ExpectedZ, Tolerance, Tolerance) ||
                NdcX < -1.0f - RangeEpsilon || NdcX > 1.0f + RangeEpsilon ||
                NdcY < -1.0f - RangeEpsilon || NdcY > 1.0f + RangeEpsilon ||
                NdcZ < ZMin - RangeEpsilon || NdcZ > ZMax + RangeEpsilon)
            {
                Obs->Failure = FAILURE_PROJECTION_RANGE;
            }
        }
    }
    return 1;
}

static void PrintProjectionCase(const TestCase *Case)
{
    printf("type=%s handedness=%s ndc=%s params=(%a, %a, %a, %a, %a, %a)",
           Case->Ints[0] ? "perspective" : "orthographic",
           Case->Ints[1] ? "LH" : "RH",
           Case->Ints[2] ? "ZO" : "NO",
           Case->Floats[0], Case->Floats[1], Case->Floats[2],
           Case->Floats[3], Case->Floats[4], Case->Floats[5]);
}

static void GenerateProjectionCase(TestCase *Case, Rng *Rng, int Class)
{
    int Perspective;
    float Scale;

    memset(Case, 0, sizeof(*Case));
    Case->Class = Class;
    Case->NumFloats = 6;
    Case->NumInts = 3;
    Perspective = RngRangeInt(Rng, 0, 1);
    Case->Ints[0] = Perspective;
    Case->Ints[1] = RngRangeInt(Rng, 0, 1);
    Case->Ints[2] = RngRangeInt(Rng, 0, 1);

    switch (Class)
    {
        case CLASS_REGULAR: Scale = 1.0f; break;
        case CLASS_NEAR_SINGULAR: Scale = 1.0f; break;
        case CLASS_TINY: Scale = 1.0e-9f; break;
        case CLASS_ZERO_VECTOR: Scale = 1.0f; break;
        case CLASS_HUGE:
        default: Scale = 1.0e9f; break;
    }

    if (Perspective)
    {
        float Near = Scale * RngRange(Rng, 0.5f, 2.0f);
        float Far = Class == CLASS_NEAR_SINGULAR
            ? Near * 1.001f
            : Near * RngRange(Rng, 1.5f, 6.5f);

        Case->Floats[0] = HMM_AngleDeg(RngRange(Rng, 20.0f, 140.0f));
        Case->Floats[1] = RngRange(Rng, 0.25f, 4.0f);
        Case->Floats[2] = 0.0f;
        Case->Floats[3] = 0.0f;
        Case->Floats[4] = Near;
        Case->Floats[5] = Far;
    }
    else
    {
        float HalfWidth = Scale * RngRange(Rng, 0.5f, 3.0f);
        float HalfHeight = Scale * RngRange(Rng, 0.5f, 3.0f);
        float Near = Scale * RngRange(Rng, 0.5f, 2.0f);
        float Far = Class == CLASS_NEAR_SINGULAR
            ? Near * 1.001f
            : Near + Scale * RngRange(Rng, 0.5f, 6.0f);

        if (Class == CLASS_ZERO_VECTOR)
        {
            Case->Floats[0] = -HalfWidth;
            Case->Floats[1] = HalfWidth;
            Case->Floats[2] = -HalfHeight;
            Case->Floats[3] = HalfHeight;
        }
        else
        {
            float Left = -HalfWidth;
            float Bottom = -HalfHeight;
            Case->Floats[0] = Left;
            Case->Floats[1] = Left + 2.0f * HalfWidth;
            Case->Floats[2] = Bottom;
            Case->Floats[3] = Bottom + 2.0f * HalfHeight;
        }
        Case->Floats[4] = Near;
        Case->Floats[5] = Far;
    }
}

/*
 * Property 5: quaternion-to-matrix consistency
 *
 * QToM4(q) * v must agree with the direct quaternion-vector rotation path, and
 * the matrix must agree with the independent axis-angle rotation path.
 * M4ToQ_RH(QToM4(q)) must also round-trip up to quaternion sign.
 */

static int EvaluateQuatMatrix(const TestCase *Case, Observation *Obs)
{
    HMM_Quat Q = HMM_Q(Case->Floats[0], Case->Floats[1], Case->Floats[2], Case->Floats[3]);
    HMM_Quat QN = NormalizeQDouble(Q);
    HMM_Vec3 V = HMM_V3(Case->Floats[4], Case->Floats[5], Case->Floats[6]);
    HMM_Mat4 M = HMM_QToM4(Q);
    HMM_Vec3 MatrixRotated = HMM_MulM4V4(M, HMM_V4(V.X, V.Y, V.Z, 0.0f)).XYZ;
    HMM_Vec3 QuatRotated = HMM_RotateV3Q(V, QN);
    HMM_Quat RoundTrip = HMM_M4ToQ_RH(M);
    HMM_Vec3 Direct;
    HMM_Mat4 DirectMatrix;
    float AxisLength = sqrtf(QN.X * QN.X + QN.Y * QN.Y + QN.Z * QN.Z);
    float VectorScale = MaxF(AbsF(V.X), MaxF(AbsF(V.Y), AbsF(V.Z)));
    float Angle;
    HMM_Vec3 Axis;
    int Column;
    int Row;
    float Tolerance = Case->Class == CLASS_NEAR_SINGULAR ? 2.0e-3f : 5.0e-4f;

    if (AxisLength == 0.0f)
    {
        Angle = 0.0f;
        Axis = HMM_V3(0.0f, 1.0f, 0.0f);
    }
    else
    {
        Axis = HMM_V3(QN.X / AxisLength, QN.Y / AxisLength, QN.Z / AxisLength);
        Angle = 2.0f * atan2f(AxisLength, QN.W);
    }
    DirectMatrix = HMM_Rotate_RH(Angle, Axis);
    Direct = HMM_V3(
        DirectMatrix.Elements[0][0] * V.X + DirectMatrix.Elements[1][0] * V.Y + DirectMatrix.Elements[2][0] * V.Z,
        DirectMatrix.Elements[0][1] * V.X + DirectMatrix.Elements[1][1] * V.Y + DirectMatrix.Elements[2][1] * V.Z,
        DirectMatrix.Elements[0][2] * V.X + DirectMatrix.Elements[1][2] * V.Y + DirectMatrix.Elements[2][2] * V.Z);

    for (Column = 0; Column < 3; ++Column)
    {
        for (Row = 0; Row < 3; ++Row)
        {
            ObserveFloat(Obs, M.Elements[Column][Row]);
            if (!NearlyEqual(M.Elements[Column][Row], DirectMatrix.Elements[Column][Row], Tolerance, Tolerance))
            {
                Obs->Failure = FAILURE_QUAT_MATRIX_DIRECT_ROTATION;
            }
        }
    }

    ObserveFloat(Obs, MatrixRotated.X);
    ObserveFloat(Obs, MatrixRotated.Y);
    ObserveFloat(Obs, MatrixRotated.Z);
    ObserveFloat(Obs, QuatRotated.X);
    ObserveFloat(Obs, QuatRotated.Y);
    ObserveFloat(Obs, QuatRotated.Z);
    ObserveFloat(Obs, Direct.X);
    ObserveFloat(Obs, Direct.Y);
    ObserveFloat(Obs, Direct.Z);
    ObserveQuat(Obs, RoundTrip);

    if (!NearlyEqualScaled(MatrixRotated.X, QuatRotated.X, Tolerance, Tolerance, VectorScale) ||
        !NearlyEqualScaled(MatrixRotated.Y, QuatRotated.Y, Tolerance, Tolerance, VectorScale) ||
        !NearlyEqualScaled(MatrixRotated.Z, QuatRotated.Z, Tolerance, Tolerance, VectorScale) ||
        !NearlyEqualScaled(MatrixRotated.X, Direct.X, Tolerance, Tolerance, VectorScale) ||
        !NearlyEqualScaled(MatrixRotated.Y, Direct.Y, Tolerance, Tolerance, VectorScale) ||
        !NearlyEqualScaled(MatrixRotated.Z, Direct.Z, Tolerance, Tolerance, VectorScale))
    {
        Obs->Failure = FAILURE_QUAT_MATRIX_ROTATION;
    }

    if (!QuatEquivalent(RoundTrip, QN, Case->Class == CLASS_NEAR_SINGULAR ? 4.0e-3f : 1.0e-3f))
    {
        Obs->Failure = FAILURE_QUAT_MATRIX_ROUNDTRIP;
    }
    return 1;
}

static void PrintQuatMatrixCase(const TestCase *Case)
{
    printf("q=(%a, %a, %a, %a) v=(%a, %a, %a)",
           Case->Floats[0], Case->Floats[1], Case->Floats[2], Case->Floats[3],
           Case->Floats[4], Case->Floats[5], Case->Floats[6]);
}

static void GenerateQuatMatrixCase(TestCase *Case, Rng *Rng, int Class)
{
    HMM_Quat Q;
    HMM_Vec3 V;
    float Scale = 1.0f;

    memset(Case, 0, sizeof(*Case));
    Case->Class = Class;
    Case->NumFloats = 7;

    switch (Class)
    {
        case CLASS_REGULAR:
        {
            Q = RandomQuat(Rng, 1.0f);
            V = HMM_V3(RngSigned(Rng), RngSigned(Rng), RngSigned(Rng));
            break;
        }
        case CLASS_NEAR_SINGULAR:
        {
            Q = QuatFromRandomAxisAngle(Rng, 3.14159265358979323846f - 1.0e-6f);
            V = HMM_V3(RngSigned(Rng), RngSigned(Rng), RngSigned(Rng));
            break;
        }
        case CLASS_TINY:
        {
            Scale = 1.0e-18f;
            Q = RandomQuat(Rng, Scale);
            V = HMM_V3(RngSigned(Rng) * Scale, RngSigned(Rng) * Scale, RngSigned(Rng) * Scale);
            break;
        }
        case CLASS_ZERO_VECTOR:
        {
            Q = RandomQuat(Rng, 1.0f);
            V = HMM_V3(0.0f, 0.0f, 0.0f);
            break;
        }
        case CLASS_HUGE:
        default:
        {
            Scale = 1.0e18f;
            Q = RandomQuat(Rng, Scale);
            V = HMM_V3(RngSigned(Rng) * Scale, RngSigned(Rng) * Scale, RngSigned(Rng) * Scale);
            break;
        }
    }

    Case->Floats[0] = Q.X;
    Case->Floats[1] = Q.Y;
    Case->Floats[2] = Q.Z;
    Case->Floats[3] = Q.W;
    Case->Floats[4] = V.X;
    Case->Floats[5] = V.Y;
    Case->Floats[6] = V.Z;
}

static int CheckCase(const Property *Property, const TestCase *Case,
                     Observation *FinalObs, int *DeterminismFailed)
{
    Observation First;
    Observation Second;

    memset(&First, 0, sizeof(First));
    memset(&Second, 0, sizeof(Second));

    if (!Property->Evaluate(Case, &First))
    {
        if (FinalObs)
        {
            *FinalObs = First;
        }
        if (DeterminismFailed)
        {
            *DeterminismFailed = 0;
        }
        return 0;
    }

    Property->Evaluate(Case, &Second);
    if (memcmp(&First, &Second, sizeof(First)) != 0)
    {
        if (FinalObs)
        {
            *FinalObs = First;
            FinalObs->Failure = FAILURE_DETERMINISM;
        }
        if (DeterminismFailed)
        {
            *DeterminismFailed = 1;
        }
        return 1;
    }

    if (FinalObs)
    {
        *FinalObs = First;
    }
    if (DeterminismFailed)
    {
        *DeterminismFailed = 0;
    }
    return First.Failure != FAILURE_NONE;
}

static double ScalarComplexity(float Value)
{
    double Magnitude;
    float IntegerPart;

    if (Value == 0.0f)
    {
        return 0.0;
    }
    Magnitude = fabs((double)Value);
    if (Magnitude == 1.0)
    {
        return 1.0;
    }
    if (Magnitude == 0.5)
    {
        return 2.0;
    }

    IntegerPart = truncf(Value);
    if (Value == IntegerPart && Magnitude < 1000000.0)
    {
        return 3.0 + Magnitude;
    }
    return 1000000.0 + log2(1.0 + Magnitude) + (double)Magnitude * 1.0e-12;
}

static int ShrinkCase(const Property *Property, TestCase *Case, long *AdditionalEvaluations)
{
    int Pass;
    int Changed = 1;
    int Index;
    static const float BasicCandidates[] = {
        0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 0.25f, -0.25f, 2.0f, -2.0f
    };

    for (Pass = 0; Pass < 64 && Changed; ++Pass)
    {
        Changed = 0;
        for (Index = 0; Index < Case->NumFloats; ++Index)
        {
            float Original = Case->Floats[Index];
            double OriginalComplexity = ScalarComplexity(Original);
            float Candidates[32];
            int CandidateCount = 0;
            int CandidateIndex;
            int ImprovedThisScalar = 0;
            int BasicIndex;
            double OriginalLog2;
            float Sign;

            for (BasicIndex = 0; BasicIndex < (int)(sizeof(BasicCandidates) / sizeof(BasicCandidates[0])); ++BasicIndex)
            {
                Candidates[CandidateCount++] = BasicCandidates[BasicIndex];
            }
            Candidates[CandidateCount++] = Original * 0.5f;
            Candidates[CandidateCount++] = Original * 0.25f;
            Candidates[CandidateCount++] = Original * 0.75f;
            Candidates[CandidateCount++] = truncf(Original);
            Candidates[CandidateCount++] = -Original;
            Candidates[CandidateCount++] = nextafterf(Original, 0.0f);
            Candidates[CandidateCount++] = nextafterf(Original, Original < 0.0f ? -INFINITY : INFINITY);

            Sign = Original < 0.0f ? -1.0f : 1.0f;
            OriginalLog2 = log2(1.0 + fabs((double)Original));
            if (isfinite(OriginalLog2))
            {
                int Exponent = (int)floor(OriginalLog2);
                float Power;
                int PowerIndex;
                for (PowerIndex = 0, Power = 1.0f;
                     PowerIndex < 30 && Power < FLT_MAX * 0.5f;
                     ++PowerIndex, Power *= 2.0f)
                {
                    if (PowerIndex <= Exponent + 1 && CandidateCount < (int)(sizeof(Candidates) / sizeof(Candidates[0])))
                    {
                        Candidates[CandidateCount++] = Sign * Power;
                    }
                }
            }

            while (!ImprovedThisScalar)
            {
                int Found = 0;
                for (CandidateIndex = 0; CandidateIndex < CandidateCount; ++CandidateIndex)
                {
                    float Candidate = Candidates[CandidateIndex];
                    if (Candidate == Original || !isfinite(Candidate))
                    {
                        continue;
                    }
                    if (ScalarComplexity(Candidate) >= OriginalComplexity)
                    {
                        continue;
                    }

                    Case->Floats[Index] = Candidate;
                    ++*AdditionalEvaluations;
                    if (CheckCase(Property, Case, NULL, NULL))
                    {
                        Original = Candidate;
                        OriginalComplexity = ScalarComplexity(Candidate);
                        Changed = 1;
                        Found = 1;
                        break;
                    }
                    Case->Floats[Index] = Original;
                }

                if (!Found)
                {
                    ImprovedThisScalar = 1;
                }
            }
            Case->Floats[Index] = Original;
        }
    }
    return Changed;
}

static void PrintClassTable(const long Counts[PROPERTY_CLASS_COUNT])
{
    int Class;
    for (Class = 0; Class < PROPERTY_CLASS_COUNT; ++Class)
    {
        printf("    %-14s %8ld\n", CLASS_NAMES[Class], Counts[Class]);
    }
}

static void PrintSeparator(void)
{
    printf("--------------------------------------------------------------------------------\n");
}

int main(int ArgumentCount, char **Arguments)
{
    static const Property Properties[] = {
        {"Orthogonality", EvaluateOrthogonality, PrintOrthogonalityCase},
        {"InverseRoundTrip", EvaluateInverse, PrintInverseCase},
        {"InterpolationEndpoints", EvaluateInterpolation, PrintInterpolationCase},
        {"ProjectionClipRange", EvaluateProjection, PrintProjectionCase},
        {"QuatToMatrixConsistency", EvaluateQuatMatrix, PrintQuatMatrixCase},
    };
    const int PropertyCount = (int)(sizeof(Properties) / sizeof(Properties[0]));
    int CasesPerProperty = PROPERTY_DEFAULT_COUNT;
    uint64_t Seed = UINT64_C(0x48414d4d41544832);
    int PropertyIndex;
    int TotalFailures = 0;
    long TotalPrimaryCases = 0;

    for (PropertyIndex = 1; PropertyIndex < ArgumentCount; ++PropertyIndex)
    {
        if (strcmp(Arguments[PropertyIndex], "--count") == 0 && PropertyIndex + 1 < ArgumentCount)
        {
            CasesPerProperty = atoi(Arguments[++PropertyIndex]);
            if (CasesPerProperty < 1)
            {
                fprintf(stderr, "--count must be positive\n");
                return 2;
            }
        }
        else if (strcmp(Arguments[PropertyIndex], "--seed") == 0 && PropertyIndex + 1 < ArgumentCount)
        {
            char *End = NULL;
            Seed = strtoull(Arguments[++PropertyIndex], &End, 0);
            if (!End || *End != '\0')
            {
                fprintf(stderr, "invalid --seed value\n");
                return 2;
            }
        }
        else
        {
            fprintf(stderr, "usage: %s [--count N] [--seed N]\n", Arguments[0]);
            return 2;
        }
    }

    printf("Handmade Math randomized property tests\n");
    printf("seed=0x%016" PRIx64 " cases_per_property=%d properties=%d\n",
           Seed, CasesPerProperty, PropertyCount);
    printf("determinism: every input is evaluated twice and compared byte-for-byte\n");
    printf("shrinking: first failure per property is reduced to a 1-minimal scalar counterexample\n");
    PrintSeparator();

    for (PropertyIndex = 0; PropertyIndex < PropertyCount; ++PropertyIndex)
    {
        const Property *Property = &Properties[PropertyIndex];
        long ClassCounts[PROPERTY_CLASS_COUNT] = {0, 0, 0, 0, 0};
        Rng Rng;
        int CasesThisProperty;
        int FirstFailureFound = 0;
        TestCase FirstFailure;
        Observation FirstFailureObservation;
        uint64_t Digest = UINT64_C(14695981039346656037);
        long AdditionalEvaluations = 0;
        int DeterminismFailure = 0;

        memset(&FirstFailure, 0, sizeof(FirstFailure));
        memset(&FirstFailureObservation, 0, sizeof(FirstFailureObservation));
        RngInit(&Rng, Seed + (uint64_t)PropertyIndex * UINT64_C(0x9e3779b97f4a7c15));

        for (CasesThisProperty = 0; CasesThisProperty < CasesPerProperty; ++CasesThisProperty)
        {
            TestCase Case;
            Observation Obs;
            int Class = CasesThisProperty % PROPERTY_CLASS_COUNT;
            int Failed;
            int DetFailed = 0;
            int FloatIndex;

            memset(&Case, 0, sizeof(Case));
            memset(&Obs, 0, sizeof(Obs));

            switch (PropertyIndex)
            {
                case 0: GenerateOrthogonalityCase(&Case, &Rng, Class); break;
                case 1: GenerateInverseCase(&Case, &Rng, Class); break;
                case 2: GenerateInterpolationCase(&Case, &Rng, Class); break;
                case 3: GenerateProjectionCase(&Case, &Rng, Class); break;
                case 4: GenerateQuatMatrixCase(&Case, &Rng, Class); break;
                default: break;
            }

            ++ClassCounts[Class];
            Failed = CheckCase(Property, &Case, &Obs, &DetFailed);
            if (DetFailed)
            {
                DeterminismFailure = 1;
            }
            if (Failed && !FirstFailureFound)
            {
                FirstFailure = Case;
                FirstFailureObservation = Obs;
                FirstFailureFound = 1;
            }

            Digest = HashInt(Digest, Failed);
            Digest = HashInt(Digest, Obs.Failure);
            for (FloatIndex = 0; FloatIndex < Obs.NumFloats; ++FloatIndex)
            {
                Digest = HashFloat(Digest, Obs.Floats[FloatIndex]);
            }
        }

        TotalPrimaryCases += (long)CasesPerProperty;
        printf("  %-24s %s  digest=0x%016" PRIx64 "\n",
               Property->Name, FirstFailureFound ? "FAIL" : "PASS", Digest);
        PrintClassTable(ClassCounts);

        if (FirstFailureFound)
        {
            int FinalFailure;
            ++TotalFailures;
            printf("    first failure: %s\n",
                   FailureName(DeterminismFailure ? FAILURE_DETERMINISM : FirstFailureObservation.Failure));
            printf("    original counterexample: ");
            Property->PrintCase(&FirstFailure);
            printf("\n");

            ShrinkCase(Property, &FirstFailure, &AdditionalEvaluations);
            {
                Observation ShrunkObs;
                memset(&ShrunkObs, 0, sizeof(ShrunkObs));
                CheckCase(Property, &FirstFailure, &ShrunkObs, &DeterminismFailure);
                FinalFailure = DeterminismFailure ? FAILURE_DETERMINISM : ShrunkObs.Failure;
            }

            printf("    shrunk counterexample (%ld extra evaluations): ",
                   AdditionalEvaluations);
            Property->PrintCase(&FirstFailure);
            printf("\n");
            printf("    shrunk failure: %s\n", FailureName(FinalFailure));
        }
        else
        {
            printf("    shrunk counterexample: none\n");
        }
        PrintSeparator();
    }

    printf("Summary: %d/%d properties passed, %ld primary cases, %ld evaluations.\n",
           PropertyCount - TotalFailures, PropertyCount,
           TotalPrimaryCases, TotalPrimaryCases * 2);
    if (TotalFailures == 0)
    {
        printf("Shrunk counterexamples: none (all properties passed).\n");
        return 0;
    }

    printf("Failed properties: %d. See the minimized counterexamples above.\n", TotalFailures);
    return 1;
}
