#include "../HandmadeTest.h"

static HMM_Mat4 InvMatrixPlaneRotation(int Row0, int Row1, float Angle)
{
    HMM_Mat4 Result = HMM_M4D(1.0f);
    float C = HMM_CosF(Angle);
    float S = HMM_SinF(Angle);

    Result.Elements[Row0][Row0] = C;
    Result.Elements[Row1][Row1] = C;
    Result.Elements[Row1][Row0] = -S;
    Result.Elements[Row0][Row1] = S;
    return Result;
}

static HMM_Mat4 InvMatrixConditionedMatrix(float Condition, int Variant)
{
    // The four singular values are {1, 0.7, 0.2, 1 / Condition}, so this
    // constructs a deterministic matrix with the requested 2-norm condition.
    float Phase = (float)Variant * 0.17f;
    HMM_Mat4 Left = HMM_MulM4(
        HMM_MulM4(InvMatrixPlaneRotation(0, 1, 0.31f + Phase),
                  InvMatrixPlaneRotation(1, 2, 0.73f - Phase)),
        InvMatrixPlaneRotation(2, 3, 1.13f + Phase * 0.5f));
    HMM_Mat4 Right = HMM_MulM4(
        HMM_MulM4(InvMatrixPlaneRotation(0, 2, 0.47f - Phase),
                  InvMatrixPlaneRotation(1, 3, 0.89f + Phase * 1.5f)),
        InvMatrixPlaneRotation(0, 3, 1.37f - Phase * 0.75f));
    HMM_Mat4 Scale = HMM_M4D(0.0f);

    Scale.Elements[0][0] = 1.0f;
    Scale.Elements[1][1] = Condition > 1.0f ? 0.7f : 1.0f;
    Scale.Elements[2][2] = Condition > 1.0f ? 0.2f : 1.0f;
    Scale.Elements[3][3] = 1.0f / Condition;

    return HMM_MulM4(HMM_MulM4(Left, Scale), HMM_TransposeM4(Right));
}

static float InvMatrixMaxAbs(HMM_Mat4 Matrix)
{
    float Result = 0.0f;
    int Column, Row;
    for (Column = 0; Column < 4; ++Column)
    {
        for (Row = 0; Row < 4; ++Row)
        {
            float Value = Matrix.Elements[Column][Row];
            float AbsValue = Value < 0.0f ? -Value : Value;
            if (AbsValue > Result)
            {
                Result = AbsValue;
            }
        }
    }
    return Result;
}

static float InvMatrixReconstructionError(HMM_Mat4 Matrix, HMM_Mat4 Inverse)
{
    // Use the normwise backward error. With float storage, the absolute error
    // of A * inverse(A) necessarily scales with the condition number.
    HMM_Mat4 Product = HMM_MulM4(Matrix, Inverse);
    float Error = 0.0f;
    int Column, Row;
    for (Column = 0; Column < 4; ++Column)
    {
        for (Row = 0; Row < 4; ++Row)
        {
            float Expected = (Column == Row) ? 1.0f : 0.0f;
            float Difference = Product.Elements[Column][Row] - Expected;
            float AbsDifference = Difference < 0.0f ? -Difference : Difference;
            if (AbsDifference > Error)
            {
                Error = AbsDifference;
            }
        }
    }

    {
        float Scale = InvMatrixMaxAbs(Matrix) * InvMatrixMaxAbs(Inverse);
        return Scale > 0.0f ? Error / Scale : Error;
    }
}

TEST(InvMatrix, Transpose)
{
    {
        HMM_Mat2 Matrix = {
            1.0f, 3.0f,
            2.0f, 4.0f,
        };
        HMM_Mat2 Expect = {
            1.0f, 2.0f,
            3.0f, 4.0f,
        };

        {
            HMM_Mat2 result = HMM_TransposeM2(Matrix);
            EXPECT_FLOAT_EQ(result.Elements[0][0], Expect.Elements[0][0]);
            EXPECT_FLOAT_EQ(result.Elements[0][1], Expect.Elements[0][1]);
            EXPECT_FLOAT_EQ(result.Elements[1][0], Expect.Elements[1][0]);
            EXPECT_FLOAT_EQ(result.Elements[1][1], Expect.Elements[1][1]);
        }
#if HANDMADE_MATH__USE_C11_GENERICS || defined(__cplusplus)
        {
            HMM_Mat2 result = HMM_Transpose(Matrix);
            EXPECT_FLOAT_EQ(result.Elements[0][0], Expect.Elements[0][0]);
            EXPECT_FLOAT_EQ(result.Elements[0][1], Expect.Elements[0][1]);
            EXPECT_FLOAT_EQ(result.Elements[1][0], Expect.Elements[1][0]);
            EXPECT_FLOAT_EQ(result.Elements[1][1], Expect.Elements[1][1]);
        }
#endif
    }

    {
        HMM_Mat3 Matrix = {
            1.0f, 4.0f, 7.0f,
            2.0f, 5.0f, 8.0f,
            3.0f, 6.0f, 9.0f,
        };
        HMM_Mat3 Expect = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f
        };

        {
            HMM_Mat3 result = HMM_TransposeM3(Matrix);
            EXPECT_FLOAT_EQ(result.Elements[0][0], Expect.Elements[0][0]);
            EXPECT_FLOAT_EQ(result.Elements[0][1], Expect.Elements[0][1]);
            EXPECT_FLOAT_EQ(result.Elements[0][2], Expect.Elements[0][2]);
            EXPECT_FLOAT_EQ(result.Elements[1][0], Expect.Elements[1][0]);
            EXPECT_FLOAT_EQ(result.Elements[1][1], Expect.Elements[1][1]);
            EXPECT_FLOAT_EQ(result.Elements[1][2], Expect.Elements[1][2]);
            EXPECT_FLOAT_EQ(result.Elements[2][0], Expect.Elements[2][0]);
            EXPECT_FLOAT_EQ(result.Elements[2][1], Expect.Elements[2][1]);
            EXPECT_FLOAT_EQ(result.Elements[2][2], Expect.Elements[2][2]);
        }
#if HANDMADE_MATH__USE_C11_GENERICS || defined(__cplusplus)
        {
            HMM_Mat3 result = HMM_Transpose(Matrix);
            EXPECT_FLOAT_EQ(result.Elements[0][0], Expect.Elements[0][0]);
            EXPECT_FLOAT_EQ(result.Elements[0][1], Expect.Elements[0][1]);
            EXPECT_FLOAT_EQ(result.Elements[0][2], Expect.Elements[0][2]);
            EXPECT_FLOAT_EQ(result.Elements[1][0], Expect.Elements[1][0]);
            EXPECT_FLOAT_EQ(result.Elements[1][1], Expect.Elements[1][1]);
            EXPECT_FLOAT_EQ(result.Elements[1][2], Expect.Elements[1][2]);
            EXPECT_FLOAT_EQ(result.Elements[2][0], Expect.Elements[2][0]);
            EXPECT_FLOAT_EQ(result.Elements[2][1], Expect.Elements[2][1]);
            EXPECT_FLOAT_EQ(result.Elements[2][2], Expect.Elements[2][2]);
        }
#endif
    }
    {
        HMM_Mat4 Matrix = {
            1.0f, 5.0f,  9.0f, 13.0f,
            2.0f, 6.0f, 10.0f, 14.0f, 
            3.0f, 7.0f, 11.0f, 15.0f,
            4.0f, 8.0f, 12.0f, 16.0f
        };
        HMM_Mat4 Expect = {
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f,
        };
        {
            HMM_Mat4 result = HMM_TransposeM4(Matrix);
            EXPECT_FLOAT_EQ(result.Elements[0][0], Expect.Elements[0][0]);
            EXPECT_FLOAT_EQ(result.Elements[0][1], Expect.Elements[0][1]);
            EXPECT_FLOAT_EQ(result.Elements[0][2], Expect.Elements[0][2]);
            EXPECT_FLOAT_EQ(result.Elements[1][0], Expect.Elements[1][0]);
            EXPECT_FLOAT_EQ(result.Elements[1][1], Expect.Elements[1][1]);
            EXPECT_FLOAT_EQ(result.Elements[1][2], Expect.Elements[1][2]);
            EXPECT_FLOAT_EQ(result.Elements[2][0], Expect.Elements[2][0]);
            EXPECT_FLOAT_EQ(result.Elements[2][1], Expect.Elements[2][1]);
            EXPECT_FLOAT_EQ(result.Elements[2][2], Expect.Elements[2][2]);
        }
#if HANDMADE_MATH__USE_C11_GENERICS || defined(__cplusplus)
        {
            HMM_Mat4 result = HMM_Transpose(Matrix);
            EXPECT_FLOAT_EQ(result.Elements[0][0], Expect.Elements[0][0]);
            EXPECT_FLOAT_EQ(result.Elements[0][1], Expect.Elements[0][1]);
            EXPECT_FLOAT_EQ(result.Elements[0][2], Expect.Elements[0][2]);
            EXPECT_FLOAT_EQ(result.Elements[1][0], Expect.Elements[1][0]);
            EXPECT_FLOAT_EQ(result.Elements[1][1], Expect.Elements[1][1]);
            EXPECT_FLOAT_EQ(result.Elements[1][2], Expect.Elements[1][2]);
            EXPECT_FLOAT_EQ(result.Elements[2][0], Expect.Elements[2][0]);
            EXPECT_FLOAT_EQ(result.Elements[2][1], Expect.Elements[2][1]);
            EXPECT_FLOAT_EQ(result.Elements[2][2], Expect.Elements[2][2]);
        }
#endif
    }

}

TEST(InvMatrix, InvGeneral)
{
    {
        HMM_Mat4 Matrix = {
            12.0f, 2.0f, 1.0f, 1.0f,
            0.0f, 0.0f, 1.0f, 1.0f,
            0.0f, 1.0f, 5.0f, 1.0f,
            11.0f, 1.0f, 0.0f, 10.0f
        };

        HMM_Mat4 Expect = HMM_M4D(1.0);
        HMM_Mat4 Inverse = HMM_InvGeneralM4(Matrix);
        HMM_Mat4 Result = HMM_MulM4(Matrix, Inverse);
        
        float Det = HMM_DeterminantM4(Matrix);
        EXPECT_FLOAT_EQ(Det, -80.0f);

        EXPECT_NEAR(Result.Elements[0][0], Expect.Elements[0][0], 0.00001f);
        EXPECT_NEAR(Result.Elements[0][1], Expect.Elements[0][1], 0.00001f);
        EXPECT_NEAR(Result.Elements[0][2], Expect.Elements[0][2], 0.00001f);
        EXPECT_NEAR(Result.Elements[0][3], Expect.Elements[0][3], 0.00001f);
        EXPECT_NEAR(Result.Elements[1][0], Expect.Elements[1][0], 0.00001f);
        EXPECT_NEAR(Result.Elements[1][1], Expect.Elements[1][1], 0.00001f);
        EXPECT_NEAR(Result.Elements[1][2], Expect.Elements[1][2], 0.00001f);
        EXPECT_NEAR(Result.Elements[1][3], Expect.Elements[1][3], 0.00001f);
        EXPECT_NEAR(Result.Elements[2][0], Expect.Elements[2][0], 0.00001f);
        EXPECT_NEAR(Result.Elements[2][1], Expect.Elements[2][1], 0.00001f);
        EXPECT_NEAR(Result.Elements[2][2], Expect.Elements[2][2], 0.00001f);
        EXPECT_NEAR(Result.Elements[2][3], Expect.Elements[2][3], 0.00001f);
        EXPECT_NEAR(Result.Elements[3][0], Expect.Elements[3][0], 0.00001f);
        EXPECT_NEAR(Result.Elements[3][1], Expect.Elements[3][1], 0.00001f);
        EXPECT_NEAR(Result.Elements[3][2], Expect.Elements[3][2], 0.00001f);
        EXPECT_NEAR(Result.Elements[3][3], Expect.Elements[3][3], 0.00001f);

#ifdef __cplusplus
        Inverse = HMM_InvGeneral(Matrix);
        Result = HMM_Mul(Matrix, Inverse);
        
        Det = HMM_Determinant(Matrix);
        EXPECT_FLOAT_EQ(Det, -80.0f);

        EXPECT_NEAR(Result.Elements[0][0], Expect.Elements[0][0], 0.00001f);
        EXPECT_NEAR(Result.Elements[0][1], Expect.Elements[0][1], 0.00001f);
        EXPECT_NEAR(Result.Elements[0][2], Expect.Elements[0][2], 0.00001f);
        EXPECT_NEAR(Result.Elements[0][3], Expect.Elements[0][3], 0.00001f);
        EXPECT_NEAR(Result.Elements[1][0], Expect.Elements[1][0], 0.00001f);
        EXPECT_NEAR(Result.Elements[1][1], Expect.Elements[1][1], 0.00001f);
        EXPECT_NEAR(Result.Elements[1][2], Expect.Elements[1][2], 0.00001f);
        EXPECT_NEAR(Result.Elements[1][3], Expect.Elements[1][3], 0.00001f);
        EXPECT_NEAR(Result.Elements[2][0], Expect.Elements[2][0], 0.00001f);
        EXPECT_NEAR(Result.Elements[2][1], Expect.Elements[2][1], 0.00001f);
        EXPECT_NEAR(Result.Elements[2][2], Expect.Elements[2][2], 0.00001f);
        EXPECT_NEAR(Result.Elements[2][3], Expect.Elements[2][3], 0.00001f);
        EXPECT_NEAR(Result.Elements[3][0], Expect.Elements[3][0], 0.00001f);
        EXPECT_NEAR(Result.Elements[3][1], Expect.Elements[3][1], 0.00001f);
        EXPECT_NEAR(Result.Elements[3][2], Expect.Elements[3][2], 0.00001f);
        EXPECT_NEAR(Result.Elements[3][3], Expect.Elements[3][3], 0.00001f);
#endif
    }

    {
        HMM_Mat3 Matrix = {
            12.0f, 2.0f, 1.0f,
            0.0f, 0.0f, 1.0f,
            0.0f, 1.0f, 5.0f
        };

        HMM_Mat3 Expect = HMM_M3D(1.0);
        HMM_Mat3 Inverse = HMM_InvGeneralM3(Matrix);
        HMM_Mat3 Result = HMM_MulM3(Matrix, Inverse);
        
        float Det = HMM_DeterminantM3(Matrix);
        EXPECT_FLOAT_EQ(Det, -12.0f);

        EXPECT_FLOAT_EQ(Result.Elements[0][0], Expect.Elements[0][0]);
        EXPECT_FLOAT_EQ(Result.Elements[0][1], Expect.Elements[0][1]);
        EXPECT_FLOAT_EQ(Result.Elements[0][2], Expect.Elements[0][2]);
        EXPECT_FLOAT_EQ(Result.Elements[1][0], Expect.Elements[1][0]);
        EXPECT_FLOAT_EQ(Result.Elements[1][1], Expect.Elements[1][1]);
        EXPECT_FLOAT_EQ(Result.Elements[1][2], Expect.Elements[1][2]);
        EXPECT_FLOAT_EQ(Result.Elements[2][0], Expect.Elements[2][0]);
        EXPECT_FLOAT_EQ(Result.Elements[2][1], Expect.Elements[2][1]);
        EXPECT_FLOAT_EQ(Result.Elements[2][2], Expect.Elements[2][2]);

#ifdef __cplusplus
        Inverse = HMM_InvGeneral(Matrix);
        Result = HMM_Mul(Matrix, Inverse);
        Det = HMM_Determinant(Matrix);
        EXPECT_FLOAT_EQ(Det, -12.0f);

        EXPECT_FLOAT_EQ(Result.Elements[0][0], Expect.Elements[0][0]);
        EXPECT_FLOAT_EQ(Result.Elements[0][1], Expect.Elements[0][1]);
        EXPECT_FLOAT_EQ(Result.Elements[0][2], Expect.Elements[0][2]);
        EXPECT_FLOAT_EQ(Result.Elements[1][0], Expect.Elements[1][0]);
        EXPECT_FLOAT_EQ(Result.Elements[1][1], Expect.Elements[1][1]);
        EXPECT_FLOAT_EQ(Result.Elements[1][2], Expect.Elements[1][2]);
        EXPECT_FLOAT_EQ(Result.Elements[2][0], Expect.Elements[2][0]);
        EXPECT_FLOAT_EQ(Result.Elements[2][1], Expect.Elements[2][1]);
        EXPECT_FLOAT_EQ(Result.Elements[2][2], Expect.Elements[2][2]);
#endif
    }

    {
        HMM_Mat2 Matrix = {
            12.0f, 2.0f, 
             1.0f, 5.0f
        };

        HMM_Mat2 Expect = HMM_M2D(1.0);
        HMM_Mat2 Inverse = HMM_InvGeneralM2(Matrix);
        HMM_Mat2 Result = HMM_MulM2(Matrix, Inverse);
        
        float Det = HMM_DeterminantM2(Matrix);
        EXPECT_FLOAT_EQ(Det, 58.0f);

        EXPECT_FLOAT_EQ(Result.Elements[0][0], Expect.Elements[0][0]);
        EXPECT_FLOAT_EQ(Result.Elements[0][1], Expect.Elements[0][1]);
        EXPECT_FLOAT_EQ(Result.Elements[1][0], Expect.Elements[1][0]);
        EXPECT_FLOAT_EQ(Result.Elements[1][1], Expect.Elements[1][1]);

#ifdef __cplusplus
        Inverse = HMM_InvGeneral(Matrix);
        Result = HMM_Mul(Matrix, Inverse);
        Det = HMM_Determinant(Matrix);
        EXPECT_FLOAT_EQ(Det, 58.0f);

        EXPECT_FLOAT_EQ(Result.Elements[0][0], Expect.Elements[0][0]);
        EXPECT_FLOAT_EQ(Result.Elements[0][1], Expect.Elements[0][1]);
        EXPECT_FLOAT_EQ(Result.Elements[1][0], Expect.Elements[1][0]);
        EXPECT_FLOAT_EQ(Result.Elements[1][1], Expect.Elements[1][1]);
#endif
    }    
}

TEST(InvMatrix, InvGeneralM4PartialPivot)
{
    const float Conditions[] = {1.0f, 10.0f, 100.0f, 1000.0f, 10000.0f};
    int ConditionIndex, Variant;

    for (ConditionIndex = 0;
         ConditionIndex < (int)(sizeof(Conditions) / sizeof(Conditions[0]));
         ++ConditionIndex)
    {
        for (Variant = 0; Variant < 16; ++Variant)
        {
            HMM_Mat4 Matrix = InvMatrixConditionedMatrix(Conditions[ConditionIndex], Variant);
            HMM_Mat4 Inverse = HMM_M4();
            HMM_Bool Success = HMM_TryInvGeneralM4(Matrix, &Inverse);

            EXPECT_TRUE(Success);
            if (Success)
            {
                EXPECT_NEAR(InvMatrixReconstructionError(Matrix, Inverse), 0.0f, 0.000001f);
            }
        }
    }

    {
        HMM_Mat4 Singular = {
            12.0f, 2.0f, 1.0f, 1.0f,
            0.0f, 0.0f, 1.0f, 1.0f,
            0.0f, 1.0f, 5.0f, 1.0f,
            11.0f, 1.0f, 0.0f, 10.0f
        };
        HMM_Mat4 Sentin = HMM_M4D(123.0f);
        HMM_Mat4 Inverse = Sentin;
        int Column;

        for (Column = 0; Column < 4; ++Column)
        {
            Singular.Elements[Column][3] =
                Singular.Elements[Column][0] + Singular.Elements[Column][1];
        }

        EXPECT_FALSE(HMM_TryInvGeneralM4(Singular, &Inverse));
        EXPECT_M4_EQ(Inverse, Sentin);
#ifdef __cplusplus
        EXPECT_FALSE(HMM_TryInvGeneral(Singular, &Inverse));
        EXPECT_M4_EQ(Inverse, Sentin);
#endif
        EXPECT_M4_EQ(HMM_InvGeneralM4(Singular), HMM_M4());
    }

    {
        HMM_Mat4 Zero = HMM_M4();
        HMM_Mat4 Inverse = HMM_M4D(123.0f);
        EXPECT_FALSE(HMM_TryInvGeneralM4(Zero, &Inverse));
    }
}

TEST(InvMatrix, InvOrthographic)
{
    {
        HMM_Mat4 Matrix = HMM_Orthographic_RH_NO(-160+100, 160+100, -90+200, 90+200, 10, 10000);
        HMM_Mat4 Inverse = HMM_InvOrthographic(Matrix);
        EXPECT_M4_EQ(HMM_MulM4(Matrix, Inverse), HMM_M4D(1.0f));
    }
    {
        HMM_Mat4 Matrix = HMM_Orthographic_RH_ZO(-160+100, 160+100, -90+200, 90+200, 10, 10000);
        HMM_Mat4 Inverse = HMM_InvOrthographic(Matrix);
        EXPECT_M4_EQ(HMM_MulM4(Matrix, Inverse), HMM_M4D(1.0f));
    }
    {
        HMM_Mat4 Matrix = HMM_Orthographic_LH_NO(-160+100, 160+100, -90+200, 90+200, 10, 10000);
        HMM_Mat4 Inverse = HMM_InvOrthographic(Matrix);
        EXPECT_M4_EQ(HMM_MulM4(Matrix, Inverse), HMM_M4D(1.0f));
    }
    {
        HMM_Mat4 Matrix = HMM_Orthographic_LH_ZO(-160+100, 160+100, -90+200, 90+200, 10, 10000);
        HMM_Mat4 Inverse = HMM_InvOrthographic(Matrix);
        EXPECT_M4_EQ(HMM_MulM4(Matrix, Inverse), HMM_M4D(1.0f));
    }
}

TEST(InvMatrix, InvPerspective)
{
    {
        HMM_Mat4 Matrix = HMM_Perspective_RH_NO(HMM_AngleDeg(120), 16.0/9.0, 10, 10000);
        HMM_Mat4 Inverse = HMM_InvPerspective_RH(Matrix);
        EXPECT_M4_EQ(HMM_MulM4(Matrix, Inverse), HMM_M4D(1.0f));
    }
    {
        HMM_Mat4 Matrix = HMM_Perspective_RH_ZO(HMM_AngleDeg(120), 16.0/9.0, 10, 10000);
        HMM_Mat4 Inverse = HMM_InvPerspective_RH(Matrix);
        EXPECT_M4_EQ(HMM_MulM4(Matrix, Inverse), HMM_M4D(1.0f));
    }
    {
        HMM_Mat4 Matrix = HMM_Perspective_LH_NO(HMM_AngleDeg(120), 16.0/9.0, 10, 10000);
        HMM_Mat4 Inverse = HMM_InvPerspective_LH(Matrix);
        EXPECT_M4_EQ(HMM_MulM4(Matrix, Inverse), HMM_M4D(1.0f));
    }
    {
        HMM_Mat4 Matrix = HMM_Perspective_LH_ZO(HMM_AngleDeg(120), 16.0/9.0, 10, 10000);
        HMM_Mat4 Inverse = HMM_InvPerspective_LH(Matrix);
        EXPECT_M4_EQ(HMM_MulM4(Matrix, Inverse), HMM_M4D(1.0f));
    }
}

TEST(InvMatrix, InvLookAt)
{
    {
        HMM_Vec3 Eye = {10.0f, 10.0f, 10.0f};
        HMM_Vec3 Center = {100.0f, 200.0f, 30.0f};
        HMM_Vec3 Up = {0.0f, 0.0f, 1.0f};
        HMM_Mat4 Matrix = HMM_LookAt_RH(Eye, Center, Up);
        HMM_Mat4 Expect = HMM_M4D(1.0f);
        HMM_Mat4 Inverse = HMM_InvLookAt(Matrix);
        HMM_Mat4 Result = HMM_MulM4(Matrix, Inverse);
        EXPECT_M4_NEAR(Result, Expect, 0.001f);
    }
    {
        HMM_Vec3 Eye = {10.0f, 10.0f, 10.0f};
        HMM_Vec3 Center = {100.0f, 200.0f, 30.0f};
        HMM_Vec3 Up = {0.0f, 0.0f, 1.0f};
        HMM_Mat4 Matrix = HMM_LookAt_LH(Eye, Center, Up);
        HMM_Mat4 Expect = HMM_M4D(1.0f);
        HMM_Mat4 Inverse = HMM_InvLookAt(Matrix);
        HMM_Mat4 Result = HMM_MulM4(Matrix, Inverse);
        EXPECT_M4_NEAR(Result, Expect, 0.001f);
    }
}

TEST(InvMatrix, InvRotate)
{
    {
        HMM_Vec3 Axis = {1.0f, -1.0f, 0.5f};
        HMM_Mat4 Matrix = HMM_Rotate_RH(HMM_AngleDeg(30), HMM_NormV3(Axis));
        HMM_Mat4 Expect = HMM_M4D(1.0f);
        HMM_Mat4 Inverse = HMM_InvRotate(Matrix);
        HMM_Mat4 Result = HMM_MulM4(Matrix, Inverse);
        EXPECT_M4_NEAR(Result, Expect, 0.001f);
    }
    {
        HMM_Vec3 Axis = {1.0f, -1.0f, 0.5f};
        HMM_Mat4 Matrix = HMM_Rotate_LH(HMM_AngleDeg(30), HMM_NormV3(Axis));
        HMM_Mat4 Expect = HMM_M4D(1.0f);
        HMM_Mat4 Inverse = HMM_InvRotate(Matrix);
        HMM_Mat4 Result = HMM_MulM4(Matrix, Inverse);
        EXPECT_M4_NEAR(Result, Expect, 0.001f);
    }
}

TEST(InvMatrix, InvScale)
{
    HMM_Vec3 Scale  = {1.0f, -1.0f, 0.5f};
    HMM_Mat4 Matrix = HMM_Scale(Scale);
    HMM_Mat4 Expect = HMM_M4D(1.0f);
    HMM_Mat4 Inverse = HMM_InvScale(Matrix);
    HMM_Mat4 Result = HMM_MulM4(Matrix, Inverse);
    EXPECT_M4_EQ(Result, Expect);
}

TEST(InvMatrix, InvTranslate)
{
    HMM_Vec3 Move  = {1.0f, -1.0f, 0.5f};
    HMM_Mat4 Matrix = HMM_Translate(Move);
    HMM_Mat4 Expect = HMM_M4D(1.0f);
    HMM_Mat4 Inverse = HMM_InvTranslate(Matrix);
    HMM_Mat4 Result = HMM_MulM4(Matrix, Inverse);
    EXPECT_M4_EQ(Result, Expect);
}
