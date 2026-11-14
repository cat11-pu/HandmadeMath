#include "../HandmadeTest.h"

/* Boundary tests for the quaternion / matrix numeric hardening. They exercise
 * tiny magnitude, subnormal, zero vector, and non-finite (NaN, inf) inputs and
 * assert that no operation crashes or leaks NaN/inf into the result. The NaN
 * and inf constants are built from their IEEE-754 bit patterns so the tests do
 * not depend on the math library exposing NAN/INFINITY. */

static float BoundaryMakeFloat(unsigned int Bits)
{
    float Result;
    memcpy(&Result, &Bits, sizeof(Result));
    return Result;
}

static float BoundaryNaN(void)
{
    return BoundaryMakeFloat(0x7FC00000u);
}

static float BoundaryInf(void)
{
    return BoundaryMakeFloat(0x7F800000u);
}

static float BoundarySubnormal(void)
{
    return BoundaryMakeFloat(0x00000001u);
}

static unsigned int BoundaryBits(float Value)
{
    unsigned int Bits;
    memcpy(&Bits, &Value, sizeof(Bits));
    return Bits;
}

static int BoundaryIsNaN(float Value)
{
    unsigned int Bits = BoundaryBits(Value);
    return (Bits & 0x7F800000u) == 0x7F800000u &&
           (Bits & 0x007FFFFFu) != 0u;
}

static int BoundaryIsInf(float Value)
{
    unsigned int Bits = BoundaryBits(Value);
    return Bits == 0x7F800000u || Bits == 0xFF800000u;
}

static int BoundaryIsPosZero(float Value)
{
    return BoundaryBits(Value) == 0x00000000u;
}

static int BoundaryQuatHasNonFinite(HMM_Quat Q)
{
    int I;
    for (I = 0; I < 4; ++I)
    {
        if (BoundaryIsNaN(Q.Elements[I]) || BoundaryIsInf(Q.Elements[I]))
        {
            return 1;
        }
    }
    return 0;
}

static int BoundaryQuatIsIdentity(HMM_Quat Q)
{
    return BoundaryIsPosZero(Q.X) && BoundaryIsPosZero(Q.Y) &&
           BoundaryIsPosZero(Q.Z) && BoundaryBits(Q.W) == BoundaryBits(1.0f);
}

static int BoundaryM4HasNonFinite(HMM_Mat4 M)
{
    int Column, Row;
    for (Column = 0; Column < 4; ++Column)
    {
        for (Row = 0; Row < 4; ++Row)
        {
            float Value = M.Elements[Column][Row];
            if (BoundaryIsNaN(Value) || BoundaryIsInf(Value))
            {
                return 1;
            }
        }
    }
    return 0;
}

static HMM_Mat4 BoundaryZeroM4(void)
{
    HMM_Mat4 M;
    int Column, Row;
    for (Column = 0; Column < 4; ++Column)
    {
        for (Row = 0; Row < 4; ++Row)
        {
            M.Elements[Column][Row] = 0.0f;
        }
    }
    return M;
}

/* ------------------------------------------------------------------ */
/* Construction path                                                  */
/* ------------------------------------------------------------------ */

TEST(BoundaryOps, ConstructZeroQuat)
{
    HMM_Quat Q = HMM_Q(0.0f, 0.0f, 0.0f, 0.0f);
    EXPECT_FLOAT_EQ(Q.X, 0.0f);
    EXPECT_FLOAT_EQ(Q.Y, 0.0f);
    EXPECT_FLOAT_EQ(Q.Z, 0.0f);
    EXPECT_FLOAT_EQ(Q.W, 0.0f);
    EXPECT_FALSE(BoundaryQuatHasNonFinite(Q));
}

TEST(BoundaryOps, ConstructSubnormalQuat)
{
    float S = BoundarySubnormal();
    HMM_Quat Q = HMM_Q(S, S, S, S);
    EXPECT_FLOAT_EQ(Q.X, S);
    EXPECT_FLOAT_EQ(Q.W, S);
    EXPECT_FALSE(BoundaryQuatHasNonFinite(Q));
}

TEST(BoundaryOps, ConstructFromNaNComponents)
{
    HMM_Quat Q = HMM_Q(BoundaryNaN(), 1.0f, 2.0f, 3.0f);
    EXPECT_TRUE(BoundaryIsNaN(Q.X));
}

TEST(BoundaryOps, ConstructFromInfComponents)
{
    HMM_Quat Q = HMM_Q(BoundaryInf(), -BoundaryInf(), 0.0f, 1.0f);
    EXPECT_TRUE(BoundaryBits(Q.X) == 0x7F800000u);
    EXPECT_TRUE(BoundaryBits(Q.Y) == 0xFF800000u);
}

TEST(BoundaryOps, AxisAngleZeroAxis)
{
    HMM_Quat Q = HMM_QFromAxisAngle_RH(HMM_V3(0.0f, 0.0f, 0.0f), 1.23f);
    EXPECT_TRUE(BoundaryQuatIsIdentity(Q));
    EXPECT_FALSE(BoundaryQuatHasNonFinite(Q));
}

TEST(BoundaryOps, AxisAngleSubnormalAxis)
{
    float S = BoundarySubnormal();
    HMM_Quat Q = HMM_QFromAxisAngle_RH(HMM_V3(S, 0.0f, 0.0f), 1.0f);
    EXPECT_FALSE(BoundaryQuatHasNonFinite(Q));
}

TEST(BoundaryOps, AxisAngleNaNAxis)
{
    HMM_Quat Q = HMM_QFromAxisAngle_RH(HMM_V3(BoundaryNaN(), 0.0f, 0.0f), 1.0f);
    EXPECT_TRUE(BoundaryQuatIsIdentity(Q));
}

TEST(BoundaryOps, AxisAngleNaNAngle)
{
    HMM_Quat Q = HMM_QFromAxisAngle_RH(HMM_V3(1.0f, 0.0f, 0.0f), BoundaryNaN());
    EXPECT_TRUE(BoundaryQuatIsIdentity(Q));
}

TEST(BoundaryOps, VecPairZeroVectors)
{
    HMM_Quat Q = HMM_QFromVecPair(HMM_V3(0.0f, 0.0f, 0.0f), HMM_V3(0.0f, 0.0f, 0.0f));
    EXPECT_FALSE(BoundaryQuatHasNonFinite(Q));
}

TEST(BoundaryOps, NormPairParallelVectors)
{
    HMM_Quat Q = HMM_QFromNormPair(HMM_V3(1.0f, 0.0f, 0.0f), HMM_V3(1.0f, 0.0f, 0.0f));
    EXPECT_TRUE(BoundaryQuatIsIdentity(Q));
}

TEST(BoundaryOps, NormPairNaNInput)
{
    HMM_Quat Q = HMM_QFromNormPair(HMM_V3(BoundaryNaN(), 0.0f, 0.0f), HMM_V3(0.0f, 1.0f, 0.0f));
    EXPECT_TRUE(BoundaryQuatIsIdentity(Q));
}

/* ------------------------------------------------------------------ */
/* Normalization path                                                 */
/* ------------------------------------------------------------------ */

TEST(BoundaryOps, NormZeroQuat)
{
    HMM_Quat Q = HMM_NormQ(HMM_Q(0.0f, 0.0f, 0.0f, 0.0f));
    EXPECT_TRUE(BoundaryQuatIsIdentity(Q));
    EXPECT_FALSE(BoundaryQuatHasNonFinite(Q));
}

TEST(BoundaryOps, NormSubnormalQuat)
{
    float S = BoundarySubnormal();
    HMM_Quat Q = HMM_NormQ(HMM_Q(S, S, S, S));
    EXPECT_FALSE(BoundaryQuatHasNonFinite(Q));
}

TEST(BoundaryOps, NormTinyMagnitudeQuat)
{
    HMM_Quat Q = HMM_NormQ(HMM_Q(1e-30f, 2e-30f, 3e-30f, 4e-30f));
    EXPECT_FALSE(BoundaryQuatHasNonFinite(Q));
}

TEST(BoundaryOps, NormNaNQuat)
{
    HMM_Quat Q = HMM_NormQ(HMM_Q(BoundaryNaN(), 1.0f, 1.0f, 1.0f));
    EXPECT_TRUE(BoundaryQuatIsIdentity(Q));
}

TEST(BoundaryOps, NormInfQuat)
{
    HMM_Quat Q = HMM_NormQ(HMM_Q(BoundaryInf(), 0.0f, 0.0f, 1.0f));
    EXPECT_TRUE(BoundaryQuatIsIdentity(Q));
}

TEST(BoundaryOps, InverseZeroQuat)
{
    HMM_Quat Q = HMM_InvQ(HMM_Q(0.0f, 0.0f, 0.0f, 0.0f));
    EXPECT_TRUE(BoundaryQuatIsIdentity(Q));
    EXPECT_FALSE(BoundaryQuatHasNonFinite(Q));
}

TEST(BoundaryOps, InverseNaNQuat)
{
    HMM_Quat Q = HMM_InvQ(HMM_Q(BoundaryNaN(), 2.0f, 3.0f, 4.0f));
    EXPECT_TRUE(BoundaryQuatIsIdentity(Q));
}

TEST(BoundaryOps, InverseSubnormalQuat)
{
    float S = BoundarySubnormal();
    HMM_Quat Q = HMM_InvQ(HMM_Q(S, 0.0f, 0.0f, 0.0f));
    EXPECT_FALSE(BoundaryQuatHasNonFinite(Q));
}

TEST(BoundaryOps, VectorNormZeroAndNaN)
{
    HMM_Vec3 Zero = HMM_NormV3(HMM_V3(0.0f, 0.0f, 0.0f));
    EXPECT_FLOAT_EQ(Zero.X, 0.0f);
    EXPECT_FLOAT_EQ(Zero.Y, 0.0f);
    EXPECT_FLOAT_EQ(Zero.Z, 0.0f);

    HMM_Vec3 Bad = HMM_NormV3(HMM_V3(BoundaryNaN(), 1.0f, 1.0f));
    EXPECT_FLOAT_EQ(Bad.X, 0.0f);
    EXPECT_FLOAT_EQ(Bad.Y, 0.0f);
    EXPECT_FLOAT_EQ(Bad.Z, 0.0f);

    EXPECT_FLOAT_EQ(HMM_LenV3(HMM_V3(BoundaryNaN(), 1.0f, 1.0f)), 0.0f);
    EXPECT_FLOAT_EQ(HMM_SqrtF(BoundaryNaN()), 0.0f);
    EXPECT_FLOAT_EQ(HMM_SqrtF(-1.0f), 0.0f);
    EXPECT_FLOAT_EQ(HMM_InvSqrtF(0.0f), 0.0f);
    EXPECT_FLOAT_EQ(HMM_InvSqrtF(BoundaryNaN()), 0.0f);
}

/* ------------------------------------------------------------------ */
/* Interpolation path                                                 */
/* ------------------------------------------------------------------ */

TEST(BoundaryOps, NLerpZeroQuats)
{
    HMM_Quat Zero = HMM_Q(0.0f, 0.0f, 0.0f, 0.0f);
    HMM_Quat Q = HMM_NLerp(Zero, 0.5f, Zero);
    EXPECT_TRUE(BoundaryQuatIsIdentity(Q));
    EXPECT_FALSE(BoundaryQuatHasNonFinite(Q));
}

TEST(BoundaryOps, NLerpNaNInput)
{
    HMM_Quat Q = HMM_NLerp(HMM_Q(BoundaryNaN(), 0.0f, 0.0f, 1.0f),
                           0.5f,
                           HMM_Q(0.0f, 0.0f, 0.0f, 1.0f));
    EXPECT_TRUE(BoundaryQuatIsIdentity(Q));
}

TEST(BoundaryOps, SLerpZeroQuats)
{
    HMM_Quat Zero = HMM_Q(0.0f, 0.0f, 0.0f, 0.0f);
    HMM_Quat Q = HMM_SLerp(Zero, 0.5f, Zero);
    EXPECT_FALSE(BoundaryQuatHasNonFinite(Q));
}

TEST(BoundaryOps, SLerpNaNInput)
{
    HMM_Quat Q = HMM_SLerp(HMM_Q(BoundaryNaN(), 0.0f, 0.0f, 1.0f),
                           0.5f,
                           HMM_Q(0.0f, 0.0f, 0.0f, 1.0f));
    EXPECT_TRUE(BoundaryQuatIsIdentity(Q));
}

TEST(BoundaryOps, SLerpSubnormalQuats)
{
    float S = BoundarySubnormal();
    HMM_Quat Tiny = HMM_Q(S, S, 0.0f, S);
    HMM_Quat Q = HMM_SLerp(Tiny, 0.5f, Tiny);
    EXPECT_FALSE(BoundaryQuatHasNonFinite(Q));
}

/* ------------------------------------------------------------------ */
/* Matrix conversion path                                             */
/* ------------------------------------------------------------------ */

TEST(BoundaryOps, QToM4ZeroQuat)
{
    HMM_Mat4 M = HMM_QToM4(HMM_Q(0.0f, 0.0f, 0.0f, 0.0f));
    EXPECT_FALSE(BoundaryM4HasNonFinite(M));
}

TEST(BoundaryOps, QToM4NaNQuat)
{
    HMM_Mat4 M = HMM_QToM4(HMM_Q(BoundaryNaN(), 0.0f, 0.0f, 1.0f));
    EXPECT_FALSE(BoundaryM4HasNonFinite(M));
}

TEST(BoundaryOps, QToM4SubnormalQuat)
{
    float S = BoundarySubnormal();
    HMM_Mat4 M = HMM_QToM4(HMM_Q(S, S, S, S));
    EXPECT_FALSE(BoundaryM4HasNonFinite(M));
}

TEST(BoundaryOps, M4ToQZeroMatrix)
{
    HMM_Mat4 M = BoundaryZeroM4();
    HMM_Quat RH = HMM_M4ToQ_RH(M);
    HMM_Quat LH = HMM_M4ToQ_LH(M);

    /* An all-zero matrix has no rotation; the key guarantee is that the result
     * stays finite (no NaN/inf); a degenerate matrix need not produce a unit
     * quaternion, and the sign of a zero bit is not part of the contract. */
    EXPECT_FALSE(BoundaryQuatHasNonFinite(RH));
    EXPECT_FALSE(BoundaryQuatHasNonFinite(LH));
}

TEST(BoundaryOps, M4ToQNaNMatrix)
{
    HMM_Mat4 M = BoundaryZeroM4();
    /* Put NaN where the trace term T is computed in every branch, so the
     * 1/sqrt(T) division is guaranteed to see a non-finite operand. */
    M.Elements[0][0] = BoundaryNaN();
    M.Elements[1][1] = BoundaryNaN();
    M.Elements[2][2] = BoundaryNaN();
    EXPECT_TRUE(BoundaryQuatIsIdentity(HMM_M4ToQ_RH(M)));
    EXPECT_TRUE(BoundaryQuatIsIdentity(HMM_M4ToQ_LH(M)));
}
