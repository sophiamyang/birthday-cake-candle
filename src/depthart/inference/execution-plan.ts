// Vendored from TypeGPU's "Monocular Light Injection" example (MIT, see ./LICENSE).
// https://github.com/software-mansion/TypeGPU/tree/7af6a9e/apps/typegpu-docs/src/examples/image-processing/monocular-light-injection
// Vector operators (a + b on vectors) are compiled by unplugin-typegpu; tsc can't type them.
// @ts-nocheck
import type { TgpuBindGroup, TgpuComputePass, TgpuComputePipeline } from 'typegpu';

interface Workgroups {
  readonly x: number;
  readonly y?: number;
  readonly z?: number;
}

export interface PreparedDispatch {
  readonly pipeline: TgpuComputePipeline;
  readonly bindGroup: TgpuBindGroup;
  readonly workgroups: Workgroups;
}

export interface OwnedGpuResource {
  destroy(): void;
}

export function recordDispatch(pass: TgpuComputePass, dispatch: PreparedDispatch): void {
  dispatch.pipeline
    .with(pass)
    .with(dispatch.bindGroup)
    .dispatchWorkgroups(
      dispatch.workgroups.x,
      dispatch.workgroups.y ?? 1,
      dispatch.workgroups.z ?? 1,
    );
}
