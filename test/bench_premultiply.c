#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

#include "../HandmadeMath.h"

#define POINT_COUNT 100000
#define ITERATIONS  200

static double NowSeconds(void)
{
    struct timespec Ts;
    clock_gettime(CLOCK_MONOTONIC, &Ts);
    return (double)Ts.tv_sec + (double)Ts.tv_nsec * 1e-9;
}

int main(void)
{
    /*
     * A realistic per-frame chain: projection * view * model, where only the
     * model matrix changes per object... here the whole chain is treated as
     * "common" and reused across all points.
     */
    HMM_Mat4 Translate = HMM_Translate(HMM_V3(1.5f, -2.0f, 3.0f));
    HMM_Mat4 Rotate = HMM_Rotate_RH(HMM_AngleDeg(37.0f), HMM_V3(0.0f, 1.0f, 0.0f));
    HMM_Mat4 Scale = HMM_Scale(HMM_V3(2.0f, 2.0f, 2.0f));

    HMM_Vec4* Points = (HMM_Vec4*)malloc(sizeof(HMM_Vec4) * POINT_COUNT);
    HMM_Vec4* OutRebuild = (HMM_Vec4*)malloc(sizeof(HMM_Vec4) * POINT_COUNT);
    HMM_Vec4* OutPremult = (HMM_Vec4*)malloc(sizeof(HMM_Vec4) * POINT_COUNT);
    if (!Points || !OutRebuild || !OutPremult)
    {
        return 1;
    }

    srand(1234);
    for (int Index = 0; Index < POINT_COUNT; ++Index)
    {
        Points[Index] = HMM_V4(
            ((float)(rand() % 2000) * 0.01f) - 10.0f,
            ((float)(rand() % 2000) * 0.01f) - 10.0f,
            ((float)(rand() % 2000) * 0.01f) - 10.0f,
            1.0f);
    }

    /*
     * The chain matrices live in volatile storage. Every inner-loop iteration
     * reloads them through a value the optimizer cannot prove loop-invariant,
     * so the per-point matrix rebuild cannot be hoisted out of the loop. This
     * models batch code whose matrices arrive across an opaque API boundary.
     */
    volatile HMM_Mat4 VTranslate = Translate;
    volatile HMM_Mat4 VRotate = Rotate;
    volatile HMM_Mat4 VScale = Scale;

    /* Baseline: rebuild the combined matrix on every point, then transform. */
    double RebuildStart = NowSeconds();
    for (int Iter = 0; Iter < ITERATIONS; ++Iter)
    {
        for (int Index = 0; Index < POINT_COUNT; ++Index)
        {
            HMM_Mat4 T = VTranslate;
            HMM_Mat4 R = VRotate;
            HMM_Mat4 S = VScale;
            HMM_Mat4 Model = HMM_MulM4(T, HMM_MulM4(R, S));
            OutRebuild[Index] = HMM_MulM4V4(Model, Points[Index]);
        }
    }
    double RebuildTime = NowSeconds() - RebuildStart;

    /* Premultiplied: fuse the chain once per batch, transform every point. */
    double PremultStart = NowSeconds();
    HMM_Mat4 Fused;
    for (int Iter = 0; Iter < ITERATIONS; ++Iter)
    {
        Fused = HMM_PremultiplyM4N(
            (HMM_Mat4[3]){ VTranslate, VRotate, VScale }, 3);
        HMM_ApplyPremultipliedM4Batch(Fused, Points, OutPremult, POINT_COUNT);
    }
    double PremultTime = NowSeconds() - PremultStart;

    /* Bit-exact check across every output point. */
    int Mismatches = 0;
    for (int Index = 0; Index < POINT_COUNT; ++Index)
    {
        if (memcmp(&OutRebuild[Index], &OutPremult[Index], sizeof(HMM_Vec4)) != 0)
        {
            ++Mismatches;
        }
    }

    double PerIterRebuild = RebuildTime / ITERATIONS;
    double PerIterPremult = PremultTime / ITERATIONS;
    double Speedup = (RebuildTime - PremultTime) / RebuildTime * 100.0;

    printf("points per batch     : %d\n", POINT_COUNT);
    printf("iterations           : %d\n", ITERATIONS);
    printf("bit-exact mismatches : %d / %d\n", Mismatches, POINT_COUNT);
    printf("rebuild total time   : %.4f s  (%.3f ms / 100k batch)\n",
        RebuildTime, PerIterRebuild * 1000.0);
    printf("premultiply total    : %.4f s  (%.3f ms / 100k batch)\n",
        PremultTime, PerIterPremult * 1000.0);
    printf("time reduction       : %.1f%%\n", Speedup);

    free(Points);
    free(OutRebuild);
    free(OutPremult);
    return Mismatches != 0;
}
