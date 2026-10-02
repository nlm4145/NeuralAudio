#pragma once

// ARM NEON kernels for 8-channel WaveNet layers (A2 standard, and the 8-channel
// arrays of the standard/lite architectures).
//
// Weight matrices are column-major (column = input channel, 8 contiguous output
// rows), so a whole 8x8 matrix fits in 16 q-registers and each input column is
// applied with lane-indexed FMAs: no broadcasts, no reloading weights per frame.
// Accumulation order matches the scalar tiled kernel (first input channel via
// multiply, then FMAs in channel order), so the conv output is bit-identical.

#if defined(__ARM_NEON) && !defined(NEURALAUDIO_DISABLE_NEON)
#define NEURALAUDIO_NEON_8CH 1
#else
#define NEURALAUDIO_NEON_8CH 0
#endif

#if NEURALAUDIO_NEON_8CH

#include <arm_neon.h>
#include <cstddef>

namespace NeuralAudio
{
	namespace Neon8
	{
		struct Matrix8x8
		{
			float32x4_t lo[8];	// rows 0-3 of column c
			float32x4_t hi[8];	// rows 4-7 of column c
		};

		static inline Matrix8x8 LoadMatrix(const float* weights)
		{
			Matrix8x8 m;

			for (int c = 0; c < 8; c++)
			{
				m.lo[c] = vld1q_f32(weights + c * 8);
				m.hi[c] = vld1q_f32(weights + c * 8 + 4);
			}

			return m;
		}

		// (a0, a1) = W * (h0, h1)
		static inline void Multiply(const Matrix8x8& w, const float32x4_t h0, const float32x4_t h1, float32x4_t& a0, float32x4_t& a1)
		{
			a0 = vmulq_laneq_f32(w.lo[0], h0, 0);
			a1 = vmulq_laneq_f32(w.hi[0], h0, 0);
			a0 = vfmaq_laneq_f32(a0, w.lo[1], h0, 1);
			a1 = vfmaq_laneq_f32(a1, w.hi[1], h0, 1);
			a0 = vfmaq_laneq_f32(a0, w.lo[2], h0, 2);
			a1 = vfmaq_laneq_f32(a1, w.hi[2], h0, 2);
			a0 = vfmaq_laneq_f32(a0, w.lo[3], h0, 3);
			a1 = vfmaq_laneq_f32(a1, w.hi[3], h0, 3);
			a0 = vfmaq_laneq_f32(a0, w.lo[4], h1, 0);
			a1 = vfmaq_laneq_f32(a1, w.hi[4], h1, 0);
			a0 = vfmaq_laneq_f32(a0, w.lo[5], h1, 1);
			a1 = vfmaq_laneq_f32(a1, w.hi[5], h1, 1);
			a0 = vfmaq_laneq_f32(a0, w.lo[6], h1, 2);
			a1 = vfmaq_laneq_f32(a1, w.hi[6], h1, 2);
			a0 = vfmaq_laneq_f32(a0, w.lo[7], h1, 3);
			a1 = vfmaq_laneq_f32(a1, w.hi[7], h1, 3);
		}

		// (a0, a1) += W * (h0, h1)
		static inline void MultiplyAccumulate(const Matrix8x8& w, const float32x4_t h0, const float32x4_t h1, float32x4_t& a0, float32x4_t& a1)
		{
			a0 = vfmaq_laneq_f32(a0, w.lo[0], h0, 0);
			a1 = vfmaq_laneq_f32(a1, w.hi[0], h0, 0);
			a0 = vfmaq_laneq_f32(a0, w.lo[1], h0, 1);
			a1 = vfmaq_laneq_f32(a1, w.hi[1], h0, 1);
			a0 = vfmaq_laneq_f32(a0, w.lo[2], h0, 2);
			a1 = vfmaq_laneq_f32(a1, w.hi[2], h0, 2);
			a0 = vfmaq_laneq_f32(a0, w.lo[3], h0, 3);
			a1 = vfmaq_laneq_f32(a1, w.hi[3], h0, 3);
			a0 = vfmaq_laneq_f32(a0, w.lo[4], h1, 0);
			a1 = vfmaq_laneq_f32(a1, w.hi[4], h1, 0);
			a0 = vfmaq_laneq_f32(a0, w.lo[5], h1, 1);
			a1 = vfmaq_laneq_f32(a1, w.hi[5], h1, 1);
			a0 = vfmaq_laneq_f32(a0, w.lo[6], h1, 2);
			a1 = vfmaq_laneq_f32(a1, w.hi[6], h1, 2);
			a0 = vfmaq_laneq_f32(a0, w.lo[7], h1, 3);
			a1 = vfmaq_laneq_f32(a1, w.hi[7], h1, 3);
		}

		// out[f] = sum_k W_k * hist[f + offsets[k]], for 8-channel frames.
		// hist points at the column of frame 0 at tap offset 0; offsets[k] <= 0.
		template <int KernelSize, int Dilation>
		static inline void DilatedConv(const float* hist, float* __restrict out, const float* const* weights, const size_t numFrames)
		{
			for (int k = 0; k < KernelSize; k++)
			{
				const Matrix8x8 w = LoadMatrix(weights[k]);
				const float* in = hist + (ptrdiff_t)(Dilation * (k + 1 - KernelSize)) * 8;

				if (k == 0)
				{
					for (size_t f = 0; f < numFrames; f++)
					{
						float32x4_t a0, a1;

						Multiply(w, vld1q_f32(in + f * 8), vld1q_f32(in + f * 8 + 4), a0, a1);

						vst1q_f32(out + f * 8, a0);
						vst1q_f32(out + f * 8 + 4, a1);
					}
				}
				else
				{
					for (size_t f = 0; f < numFrames; f++)
					{
						float32x4_t a0 = vld1q_f32(out + f * 8);
						float32x4_t a1 = vld1q_f32(out + f * 8 + 4);

						MultiplyAccumulate(w, vld1q_f32(in + f * 8), vld1q_f32(in + f * 8 + 4), a0, a1);

						vst1q_f32(out + f * 8, a0);
						vst1q_f32(out + f * 8 + 4, a1);
					}
				}
			}
		}

		// x > 0 ? x : 0.01 * x. For a slope in (0, 1) this is max(x, 0.01 * x) for
		// every finite x, signed zeros included (both operands share x's sign).
		static inline float32x4_t LeakyReLU(const float32x4_t x)
		{
			return vmaxq_f32(x, vmulq_n_f32(x, 0.01f));
		}
	}
}

#endif
