# 附录 E. EvoTrace V4 的上下文与提示词

## E.1 公共生成模板

~~~~text
[Algorithm-design task]
{task_description}

Design the computation performed by the following function to improve the complete solver's objective. Use the available inputs and the role of its returned result in the solver.

[Function interface]
```python
{function_template}
```

[Context]
{context}

[Instruction]
{instruction}

[Output]
Return exactly one valid JSON object with three string fields, in this order:
- "analysis": {analysis_instruction} Use up to 300 words.
- "code": Provide the complete implementation using only the inputs in [Function interface], including necessary imports and helper definitions.
- "design": In about 120 words, describe for another algorithm designer the main computation the code performs and its key parameters. Describe the implementation, not expected performance.
Keep "design" faithful to "code". Preserve the required function signature and satisfy the task constraints and input/output requirements.
Return only the JSON object, without Markdown fences or additional text.
~~~~

初始化和三类演化算子共用以下 `{analysis_instruction}`。

~~~~text
Use the structure of the problem to work out a concrete computation. Which choices would it favor, and why could they improve the final solution? Draw on earlier results when useful.
~~~~

## E.2 生成上下文

**表 E.1. 各类生成调用的上下文顺序。**

| 调用 | 上下文顺序 |
| --- | --- |
| 初始化（Init） | `[Example Algorithm]` 参考实现 → `[Existing Approaches]` 已有初始算法的成绩与设计 → `[Failed Attempts]` 失败程序及反馈 |
| 精化（Refine） | `[Development of the Current Algorithm]` 主父代的形成路径 → `[Attempts from the Current Algorithm]` 从主父代出发的尝试 → `[Previous Retained Implementation]` 最近保留前驱（有空间时） → `[Current Algorithm]` 主父代的成绩与源码 |
| 交叉（Crossover） | `[Development of the Current Algorithm]` 主父代的形成路径 → `[Attempts from the Current Algorithm]` 从主父代出发的尝试 → `[Other Algorithm]` 参考算法卡片 → `[Current Algorithm]` 主父代的成绩与源码 |
| 探索（Explore） | `[Current Algorithm]` 主父代卡片 → `[Other Existing Approaches]` 其他代表的卡片 |

形成路径和尝试列表按算子目标或路径收缩情况标注来由，并展示实测结果、设计简述和已有经验。精化与交叉在形成路径的每条原始边上展示对应子代生成时自述的 `analysis`，其中的引用属于当时的上下文。收缩边展示该段路径已有的反思。个体卡片标明候选自述和局部反思的来源，并展示实测成绩和实现。精化中，若前驱与当前算法的展示源码相同，前驱位置只保留成绩与相同说明。

## E.3 生成指令

### E.3.1 初始化（Initialization）

**首次获得可行算法前。** 下列文本对应提供了示例的任务。任务未提供示例时，省去提及示例的那句话。

~~~~text
Design an effective algorithm for this task. Consider how the returned result affects the complete solution and choose a computational approach suited to that role. The example illustrates an implementation, not a structure to preserve. Implement the specified function.
~~~~

**已有可行初始算法时。**

~~~~text
Develop an effective algorithm with a different main computation from the ideas already represented. Seek a different way to compute the result, rather than another weighting of the same local features. You can change the representation, the information used, or the computational procedure. Implement the chosen approach as a complete algorithm.
~~~~

**展示失败尝试时追加。**

~~~~text
Use the failed attempts and evaluation feedback to inform your implementation.
~~~~

### E.3.2 精化（Refine）

~~~~text
Improve the current algorithm. Choose the modification you judge most promising for this task.
~~~~

### E.3.3 交叉（Crossover）

~~~~text
Develop the current algorithm by combining ideas from the two implementations where they work well together.
~~~~

### E.3.4 探索（Explore）

~~~~text
Develop a different way to compute the result. Choose an approach that looks promising for this task.
~~~~

## E.4 反思（Reflection）

~~~~text
{problem}

[Algorithms]
{situation}

{records}

[Task]
{question}
Compare what each attempt intended, what its code computed, and how it performed. What would a designer miss from the current implementation and its score alone? Ground each judgment in the outcomes shown and preserve comparisons that matter for the next design decision. When several changes were tested together, report their combined outcome and treat individual contributions as hypotheses unless another comparison isolates them. If an idea still looks worth developing after a poor result, explain why in light of what was actually tested.{merge}

[Output]
Return a JSON object with two strings. In "analysis", briefly compare the attempts. In "experience", state the useful design insight as briefly as it warrants. Describe the computations in words, omitting algorithm numbers and parameter values. If there is nothing to add or revise, return an empty string for "experience".
~~~~

`{problem}` 给出任务、目标定义和函数接口。`{records}` 汇集源码、设计简述、实测结果或失败信息，以及已有经验。原始边附上相应尝试的生成前分析。已收缩的边以已有的路径经验参与下一次反思。路径收缩区分主路径与侧枝。子树剪枝将尝试结果与起点算法比较。

| 区域 | `{situation}` |
| --- | --- |
| 从算法节点开始的路径收缩 | Algorithm {v} was developed from Algorithm {u} along {path}; overall {overall_outcome}. Side attempts along the way are included. |
| 从虚拟根开始的路径收缩 | Algorithm {v} was developed along {path}, starting from a design made from scratch. Side attempts along the way are included. |
| 算法节点下的子树剪枝 | The algorithms below are attempts that started from Algorithm {p}, directly or from one another. |

**路径收缩的反思问题。**

~~~~text
Reflect, as an algorithm designer, on how Algorithm {v} was developed. Which ideas shaped it, and did any intermediate or side approach work better than the final design?
~~~~

**子树剪枝的反思问题。**

~~~~text
What did the branches from Algorithm {p} try? What do their outcomes show about the implementations tested?
~~~~

**已有经验时追加。**

~~~~text
Earlier experience summarizes previous attempts. Repeating it adds no evidence. If these attempts change or extend it, write the updated experience, keeping earlier insights that still apply. An empty experience keeps the earlier notes.
~~~~

## E.5 修复（Repair）

| 修复类型 | 上下文顺序 |
| --- | --- |
| 格式修复 | `[Original Response]` 原始回复 → `[Parsing Feedback]` 解析反馈 |
| 报错或超时修复 | `[Current Algorithm]` 失败候选的设计与源码 → `[Failure Feedback]` 异常或超时信息 |

### E.5.1 格式修复

~~~~text
Convert the original response into the JSON object specified in [Output], using the parsing feedback to correct its structure, required fields, and string escaping.
Preserve the existing algorithmic idea and computational logic.
Complete any missing or truncated content consistently with that approach and the task requirements.
~~~~

`{analysis_instruction}`

~~~~text
Keep the original rationale if present. Otherwise briefly state the intended computation.
~~~~

### E.5.2 报错修复

~~~~text
Trace the reported exception or invalid result to its cause in the code, using the design to understand the intended algorithm.
Correct the faulty computations or data handling so the function executes correctly and satisfies the task requirements, while retaining the intended core computation.
~~~~

### E.5.3 超时修复

~~~~text
Make the algorithm finish within the reported time limit. Reduce the cost of its core computation, using an approximation when needed, while retaining the intended algorithmic idea.
~~~~

报错与超时修复共用以下 `{analysis_instruction}`。

~~~~text
Explain the correction for the reported failure.
~~~~
