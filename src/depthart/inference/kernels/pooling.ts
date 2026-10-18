// Vendored from TypeGPU's "Monocular Light Injection" example (MIT, see ./LICENSE).
// https://github.com/software-mansion/TypeGPU/tree/7af6a9e/apps/typegpu-docs/src/examples/image-processing/monocular-light-injection
// Vector operators (a + b on vectors) are compiled by unplugin-typegpu; tsc can't type them.
// @ts-nocheck
import { d, tgpu } from 'typegpu';
import { blockedElement, hwc4Index, maskPaddedChannels } from './helpers.ts';
import { poolLayout } from './layouts.ts';
import { DEPTH_KERNEL_WORKGROUP_SIZE } from './types.ts';

/** Fixed-window average pool. DepthART uses non-padded 8/4/2 windows and equal strides */
export const averagePoolKernel = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [DEPTH_KERNEL_WORKGROUP_SIZE],
})(({ gid }) => {
  'use gpu';
  const index = gid.x;
  const params = poolLayout.$.params;
  if (index >= params.elementCount) {
    return;
  }

  const output = blockedElement(index, params.outputWidth, params.channelBlocks);
  const inputOrigin = d.vec2u(output.x * params.strideX, output.y * params.strideY);
  let sum = d.vec4f(0);

  for (let ky = d.u32(0); ky < params.windowHeight; ky += 1) {
    for (let kx = d.u32(0); kx < params.windowWidth; kx += 1) {
      const inputX = inputOrigin.x + kx;
      const inputY = inputOrigin.y + ky;
      if (inputX < params.inputWidth && inputY < params.inputHeight) {
        sum +=
          poolLayout.$.src[
            hwc4Index(inputY, inputX, output.z, params.inputWidth, params.channelBlocks)
          ];
      }
    }
  }

  const divisor = d.f32(params.windowWidth * params.windowHeight);
  poolLayout.$.dst[index] = maskPaddedChannels(sum / divisor, output.z, params.logicalChannels);
});
