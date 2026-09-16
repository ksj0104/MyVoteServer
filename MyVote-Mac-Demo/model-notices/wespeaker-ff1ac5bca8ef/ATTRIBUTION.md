# Downloaded research assets

- `voxceleb_resnet34.onnx`: WeSpeaker authors, official repository
  <https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34>, revision
  `ff1ac5bca8ef11e90662b879aa923979e0bd277b`. Downloaded unchanged.
- Upstream pretrained-model documentation identifies VoxCeleb weights as CC BY 4.0:
  <https://github.com/wenet-e2e/wespeaker/blob/45941e7cba2c3ea99e232d02bedf617fc71b0dad/docs/pretrained.md>.
  That Hugging Face model card instead tags Apache-2.0. Both statements are retained
  in `manifest.json`; their difference has not been resolved as redistribution approval.
  The upstream code license is separately saved as `upstream-code-LICENSE`.
- `Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav`: SRI International and Lab41,
  In-Q-Tel; Richey et al. (2018), *Voices Obscured in Complex Environmental Settings
  (VOICES) corpus*, <https://arxiv.org/abs/1804.05053>.
  Dataset: <https://iqtlabs.github.io/voices/>.
  License: CC BY 4.0, <https://creativecommons.org/licenses/by/4.0/>.
  Retrieved from the official TorchAudio tutorial asset URL recorded in the manifest.
  The WAV file is unchanged; the smoke reads its first three seconds in memory.

No endorsement of MyVote by the asset creators is implied. Checksums, download
URLs, pinned revisions and CPU-wheel origins are recorded in `manifest.json`.
