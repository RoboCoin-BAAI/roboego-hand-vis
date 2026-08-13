# Stage 1 Action Taxonomy

This document records the initial atomic action vocabulary for selecting useful
egocentric video segments after Stage 1 Qwen frame annotation.

Stage 1 Qwen output remains descriptive. Atomic actions are extracted later from
`activity`, `hands_visible`, `interaction_objects`, `interaction_object_attributes`,
and `attention_objects`.

## Atomic Skill Actions

| Chinese | English |
| --- | --- |
| 移动 | move |
| 抓取 | grasp |
| 释放 | release |
| 放置 | place |
| 按压 | press |
| 旋转 | rotate |
| 推 | push |
| 拉 | pull |
| 翻转 | flip |
| 切割 | cut |
| 插入 | insert |
| 拔出 | pull out |
| 搅拌 | stir |
| 擦拭 | wipe |
| 折叠 | fold |
| 打开 | open |
| 关闭 | close |
| 对齐 | align |
| 倾倒 | pour |
| 测量 | measure |
| 分拣 | sort |
| 堆叠 | stack |
| 装载 | load |
| 卸下 | unload |
| 拾取 | pick |
| 下放 | lower |
| 提起 | lift |
| 传递 | pass |
| 移交 | handover |
| 拆分 | pull apart |
| 合并 | push together |
| 展开 | unfold |
| 开盖 | uncap |
| 取出 | take out |
| 捏取 | pinch |
| 夹紧 | clip |
| 按按钮 | press button |
| 扫描 | scan |
| 操作 | operate |
| 放入 | put into |
| 取自 | take from |
| 放回 | put back |
| 支撑 | support |

## Hand State / Pre-Action Cues

These actions are still hand-related, but they are weaker than manipulation
skills. They can help expand segment boundaries or detect pre-action context.

| Chinese | English | Notes |
| --- | --- | --- |
| 持有 | hold | Ongoing possession, e.g. holding a phone. Do not merge with grasp. |
| 携带 | carry | Holding while moving; often context unless paired with another operation. |
| 接触 | touch | Contact without a clear manipulation verb. |
| 靠近/伸手 | reach | Pre-operation cue. |
| 指向 | point | Attention cue, may become useful when paired with object selection. |
| 手势 | gesture | Hand motion without a concrete operated object. |

## Non-Hand Context Cues Excluded From `frame_actions`

These can remain in Qwen's `activity`, `scene`, `task_context`, or
`attention_objects`, but are not extracted as hand actions:

| Chinese | English | Reason |
| --- | --- | --- |
| 查看/注视 | look at | Visual attention, not hand motion. |
| 检查 | inspect | Often visual; extract only when paired with a hand verb such as hold/rotate. |
| 阅读 | read | Visual/text attention. |
| 选择 | select | Semantic intent unless paired with reach/pick/grasp. |
| 支付 | pay | Retail/payment context; use scan/press/operate for hand action. |
| 结账 | checkout | Retail context. |
| 走路 | walk | Wearer locomotion. |
| 站立 | stand | Posture/context only. |
| 等待 | wait | Idle/context only. |
| 交谈 | talk | Social/context cue. |

## Verb Mapping Notes

- `holds`, `holding`, `held` map to `hold`.
- `grasp`, `grab` map to `grasp`.
- `carry` maps to `carry`; do not merge it into `hold`.
- `pick up` maps to `pick`.
- `put down` maps to `place`.
- `put into` binds preferentially to the destination container.
- `take from` binds preferentially to the source container.
- `scan` and `operate` are hand actions when bound to an object.
- `pay` and `checkout` are context cues and are not extracted as `frame_actions`.
- `walk` does not map to `move`; `move` is reserved for moving an object.

## Binding Policy Draft

For each frame, first extract candidate actions from `activity`. Then bind each
action to objects with explicit evidence:

1. If the activity text contains both an action verb and an object name from
   `interaction_object_attributes.name`, bind the action to that object.
2. If only one interaction object exists, bind hand actions in the activity to
   that object.
3. If multiple objects exist, use `hand_side` and hand mentions in `activity` or
   `visual_description` to bind actions per hand.
4. If the action is attention-only, such as looking at or pointing at an object,
   allow binding to `attention_objects`.
5. If no object can be identified with confidence, keep the action with
   `object_name=unknown` instead of guessing.

Do not permanently assign `object_id` at this stage. Persistent IDs are added
later after segmentation, mesh extraction, and object library matching.
