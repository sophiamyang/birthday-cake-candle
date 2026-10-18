// Vendored from TypeGPU's "Monocular Light Injection" example (MIT, see ./LICENSE).
// https://github.com/software-mansion/TypeGPU/tree/7af6a9e/apps/typegpu-docs/src/examples/image-processing/monocular-light-injection
// Vector operators (a + b on vectors) are compiled by unplugin-typegpu; tsc can't type them.
// @ts-nocheck
import { d, std, tgpu } from 'typegpu';
import { channelConcatLayout, channelSplitLayout } from './layouts.ts';
import { DEPTH_KERNEL_WORKGROUP_SIZE } from './types.ts';

export const channelSplitKernel = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [DEPTH_KERNEL_WORKGROUP_SIZE],
})(({ gid }) => {
  'use gpu';
  const index = gid.x;
  const params = channelSplitLayout.$.params;
  if (index >= params.elementCount) {
    return;
  }

  const sourceBlock = index % params.totalChannelBlocks;
  const pixel = std.intdiv(index, params.totalChannelBlocks);
  if (sourceBlock < params.lowChannelBlocks) {
    const destinationIndex = pixel * params.lowChannelBlocks + sourceBlock;
    channelSplitLayout.$.lowDst[destinationIndex] = d.vec4f(channelSplitLayout.$.src[index]);
  } else {
    const destinationIndex =
      pixel * params.highChannelBlocks + sourceBlock - params.lowChannelBlocks;
    channelSplitLayout.$.highDst[destinationIndex] = d.vec4f(channelSplitLayout.$.src[index]);
  }
});

export const channelConcatKernel = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [DEPTH_KERNEL_WORKGROUP_SIZE],
})(({ gid }) => {
  'use gpu';
  const index = gid.x;
  const params = channelConcatLayout.$.params;
  if (index >= params.elementCount) {
    return;
  }

  const destinationBlock = index % params.totalChannelBlocks;
  const pixel = std.intdiv(index, params.totalChannelBlocks);
  if (destinationBlock < params.lowChannelBlocks) {
    const sourceIndex = pixel * params.lowChannelBlocks + destinationBlock;
    channelConcatLayout.$.dst[index] = d.vec4f(channelConcatLayout.$.lowSrc[sourceIndex]);
  } else {
    const sourceIndex =
      pixel * params.highChannelBlocks + destinationBlock - params.lowChannelBlocks;
    channelConcatLayout.$.dst[index] = d.vec4f(channelConcatLayout.$.highSrc[sourceIndex]);
  }
});
