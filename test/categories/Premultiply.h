#include <string.h>

static HMM_Mat4 PremultiplyTestMat4(int Seed)
{
    HMM_Mat4 Matrix = HMM_M4();
    for (int Column = 0; Column < 4; ++Column)
    {
        for (int Row = 0; Row < 4; ++Row)
        {
            Matrix.Elements[Column][Row] =
                (float)(((Column * 4 + Row + Seed) % 7) - 3) * 0.5f;
        }
    }
    Matrix.Elements[3][3] = 1.0f;
    return Matrix;
}

static HMM_Vec4 PremultiplyTestVec4(int Index)
{
    return HMM_V4(
        (float)((Index * 13) % 17) - 8.0f,
        (float)((Index * 7) % 11) - 5.0f,
        (float)((Index * 5) % 13) - 6.0f,
        1.0f);
}

TEST(Premultiply, TwoMatrices)
{
    HMM_Mat4 First = PremultiplyTestMat4(1);
    HMM_Mat4 Second = PremultiplyTestMat4(4);

    HMM_Mat4 Expected = HMM_MulM4(First, Second);
    HMM_Mat4 Actual = HMM_PremultiplyM4(First, Second);

    EXPECT_M4_EQ(Actual, Expected);
    EXPECT_TRUE(memcmp(&Actual, &Expected, sizeof(HMM_Mat4)) == 0);
}

TEST(Premultiply, ChainN)
{
    HMM_Mat4 Chain[3];
    Chain[0] = PremultiplyTestMat4(1);
    Chain[1] = PremultiplyTestMat4(4);
    Chain[2] = PremultiplyTestMat4(2);

    HMM_Mat4 Expected = HMM_MulM4(HMM_MulM4(Chain[0], Chain[1]), Chain[2]);
    HMM_Mat4 Actual = HMM_PremultiplyM4N(Chain, 3);

    EXPECT_M4_EQ(Actual, Expected);
    EXPECT_TRUE(memcmp(&Actual, &Expected, sizeof(HMM_Mat4)) == 0);

    HMM_Mat4 Identity = HMM_PremultiplyM4N(Chain, 0);
    EXPECT_M4_EQ(Identity, HMM_M4D(1.0f));
}

TEST(Premultiply, BitExactPerPoint)
{
    HMM_Mat4 First = PremultiplyTestMat4(1);
    HMM_Mat4 Second = PremultiplyTestMat4(4);
    HMM_Mat4 Third = PremultiplyTestMat4(2);

    HMM_Mat4 Fused = HMM_PremultiplyM4N(
        (HMM_Mat4[3]){ First, Second, Third }, 3);

    for (int Index = 0; Index < 64; ++Index)
    {
        HMM_Vec4 Point = PremultiplyTestVec4(Index);

        HMM_Vec4 Chained = HMM_MulM4V4(
            HMM_MulM4(HMM_MulM4(First, Second), Third), Point);
        HMM_Vec4 FusedResult = HMM_ApplyPremultipliedM4(Fused, Point);

        EXPECT_V4_EQ(FusedResult, Chained);
        EXPECT_TRUE(memcmp(&FusedResult, &Chained, sizeof(HMM_Vec4)) == 0);
    }
}

TEST(Premultiply, Vec3Convenience)
{
    HMM_Mat4 First = PremultiplyTestMat4(1);
    HMM_Mat4 Second = PremultiplyTestMat4(4);
    HMM_Mat4 Fused = HMM_PremultiplyM4(First, Second);

    HMM_Vec3 Point = HMM_V3(2.5f, -1.5f, 3.0f);
    HMM_Vec4 Reference4 = HMM_MulM4V4(Fused, HMM_V4(Point.X, Point.Y, Point.Z, 1.0f));
    HMM_Vec3 Actual = HMM_ApplyPremultipliedM4V3(Fused, Point);

    EXPECT_FLOAT_EQ(Actual.X, Reference4.X);
    EXPECT_FLOAT_EQ(Actual.Y, Reference4.Y);
    EXPECT_FLOAT_EQ(Actual.Z, Reference4.Z);
}

TEST(Premultiply, Batch)
{
    HMM_Mat4 First = PremultiplyTestMat4(1);
    HMM_Mat4 Second = PremultiplyTestMat4(4);
    HMM_Mat4 Fused = HMM_PremultiplyM4(First, Second);

    enum { Count = 32 };
    HMM_Vec4 Points[Count];
    HMM_Vec4 Output[Count];
    for (int Index = 0; Index < Count; ++Index)
    {
        Points[Index] = PremultiplyTestVec4(Index);
    }

    HMM_ApplyPremultipliedM4Batch(Fused, Points, Output, Count);

    for (int Index = 0; Index < Count; ++Index)
    {
        HMM_Vec4 Expected = HMM_MulM4V4(
            HMM_MulM4(First, Second), Points[Index]);
        EXPECT_V4_EQ(Output[Index], Expected);
        EXPECT_TRUE(memcmp(&Output[Index], &Expected, sizeof(HMM_Vec4)) == 0);
    }
}
