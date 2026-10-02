#pragma once

#include <algorithm>
#include <vector>
#include "NeuralModel.h"
#include "NeuralModelImpl.h"

namespace NeuralAudio
{
	// Runs a WaveNet at an integer multiple F of its native sample rate.
	//
	// Scaling every dilation by F (the classic way to "oversample" a WaveNet)
	// makes each convolution tap read only samples of the same phase (t mod F),
	// while everything else in the network (rechannel, mixin, 1x1, activations,
	// kernel-1 heads) is per-sample. The dilation-scaled network is therefore
	// exactly F independent copies of the native-rate network, copy p running on
	// samples p, p+F, p+2F, ... This class runs it that way, so each copy can use
	// the fast native-rate kernels (static templates / A2) and keeps a history
	// buffer F times smaller.
	//
	// Process() is real-time safe: no allocation, any block length, and phase
	// alignment persists across calls.
	class PolyphaseModel : public NeuralModelImpl
	{
		public:
			explicit PolyphaseModel(std::vector<NeuralModelImpl*>&& phaseModels)
				: phases(std::move(phaseModels))
			{
				numPhases = phases.size();
				phaseOffsets.resize(numPhases + 1);
			}

			~PolyphaseModel()
			{
				for (auto* model : phases)
				{
					delete model;
				}
			}

			size_t GetNumPhases() const
			{
				return numPhases;
			}

			EModelLoadMode GetLoadMode() override
			{
				return phases[0]->GetLoadMode();
			}

			bool HasQualityScaling() override
			{
				return phases[0]->HasQualityScaling();
			}

			float GetQualityScaleFactor() override
			{
				return phases[0]->GetQualityScaleFactor();
			}

			bool IsQualityChangeRealtimeSafe(float newScaleFactor) override
			{
				for (auto* model : phases)
				{
					if (!model->IsQualityChangeRealtimeSafe(newScaleFactor))
						return false;
				}

				return true;
			}

			void SetQualityScaleFactor(float scaleFactor) override
			{
				for (auto* model : phases)
				{
					model->SetQualityScaleFactor(scaleFactor);
				}
			}

			bool IsStatic() override
			{
				return phases[0]->IsStatic();
			}

			void SetMaxAudioBufferSize(const int maxSize) override
			{
				const size_t size = (size_t)std::max(1, maxSize);

				if (scratch.size() < size)
					scratch.resize(size);

				const int perPhase = (int)((size + numPhases - 1) / numPhases) + 1;

				for (auto* model : phases)
				{
					model->SetMaxAudioBufferSize(perPhase);
				}
			}

			void SetAudioInputLevelDBu(float audioDBu) override
			{
				NeuralModel::SetAudioInputLevelDBu(audioDBu);

				for (auto* model : phases)
				{
					model->SetAudioInputLevelDBu(audioDBu);
				}
			}

			float GetAudioInputLevelDBu() override
			{
				return phases[0]->GetAudioInputLevelDBu();
			}

			float GetRecommendedInputDBAdjustment() override
			{
				return phases[0]->GetRecommendedInputDBAdjustment();
			}

			float GetRecommendedOutputDBAdjustment() override
			{
				return phases[0]->GetRecommendedOutputDBAdjustment();
			}

			float GetSampleRate() override
			{
				return phases[0]->GetSampleRate();
			}

			int GetReceptiveFieldSize() override
			{
				const int size = phases[0]->GetReceptiveFieldSize();

				return (size < 0) ? size : size * (int)numPhases;
			}

			std::string GetModelVersion() override
			{
				return phases[0]->GetModelVersion();
			}

			std::string GetMetadata(const std::string& fieldName) override
			{
				return phases[0]->GetMetadata(fieldName);
			}

			void Process(float* input, float* output, size_t numSamples) override
			{
				const size_t maxChunk = scratch.size();

				if (maxChunk == 0)
					return;

				while (numSamples > 0)
				{
					const size_t chunk = std::min(numSamples, maxChunk);
					float* buffer = scratch.data();

					// Deinterleave: sample i belongs to phase (nextPhase + i) % F.
					// Phase p's sub-block is contiguous at phaseOffsets[p].
					size_t pos = 0;

					for (size_t p = 0; p < numPhases; p++)
					{
						phaseOffsets[p] = pos;

						for (size_t i = (p + numPhases - nextPhase) % numPhases; i < chunk; i += numPhases)
						{
							buffer[pos++] = input[i];
						}
					}

					phaseOffsets[numPhases] = pos;

					for (size_t p = 0; p < numPhases; p++)
					{
						const size_t count = phaseOffsets[p + 1] - phaseOffsets[p];

						if (count > 0)
							phases[p]->Process(buffer + phaseOffsets[p], buffer + phaseOffsets[p], count);
					}

					// Interleave back. Input was fully copied first, so in-place is fine.
					for (size_t p = 0; p < numPhases; p++)
					{
						pos = phaseOffsets[p];

						for (size_t i = (p + numPhases - nextPhase) % numPhases; i < chunk; i += numPhases)
						{
							output[i] = buffer[pos++];
						}
					}

					nextPhase = (nextPhase + chunk) % numPhases;
					input += chunk;
					output += chunk;
					numSamples -= chunk;
				}
			}

			void Prewarm() override
			{
				for (auto* model : phases)
				{
					model->Prewarm();
					model->SetHadInitialPrewarm();
				}
			}

		private:
			std::vector<NeuralModelImpl*> phases;
			std::vector<size_t> phaseOffsets;
			std::vector<float> scratch;
			size_t numPhases = 1;
			size_t nextPhase = 0;	// phase of the next input sample
	};
}
