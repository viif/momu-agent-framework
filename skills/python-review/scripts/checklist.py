BASE_CHECKLIST = [
    "检查命名是否表达意图，避免缩写和歧义名称",
    "检查函数职责是否单一，是否存在过长函数或过深嵌套",
    "检查异常是否只在边界处理，避免静默吞错",
    "检查输入输出边界是否清晰，是否存在隐式副作用",
]

FOCUS_CHECKLISTS = {
    "tooling": [
        "检查是否优先复用现有工具、基类或注册机制",
        "检查新增能力是否符合现有接口约定而不是另起一套模式",
    ],
    "async": [
        "检查 async/await 使用是否一致，是否遗漏 await",
        "检查是否在不合适的位置阻塞事件循环",
    ],
    "security": [
        "检查是否直接拼接命令、路径或外部输入，避免注入风险",
        "检查敏感信息是否被写入日志、异常或调试输出",
    ],
    "testing": [
        "检查关键分支、异常路径和边界条件是否可测试",
        "检查改动是否依赖难以验证的隐式行为",
    ],
}


def review_checklist(*focus_areas: str) -> list[str]:
    checklist = list(BASE_CHECKLIST)

    for area in focus_areas:
        items = FOCUS_CHECKLISTS.get(area)
        if items:
            checklist.extend(items)

    return checklist
