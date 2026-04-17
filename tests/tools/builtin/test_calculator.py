import pytest

from momu_agent.core.exceptions import ToolException
from momu_agent.tools import CalculatorTool


class TestCalculatorTool:
    """测试 CalculatorTool 类"""

    @pytest.fixture
    def calculator(self):
        """创建计算器工具实例"""
        return CalculatorTool()

    @pytest.mark.asyncio
    async def test_basic_arithmetic(self, calculator):
        """测试加减乘除"""
        assert await calculator.run({"input": "2 + 3"}) == "5"
        assert await calculator.run({"input": "5 - 2"}) == "3"
        assert await calculator.run({"input": "3 * 4"}) == "12"
        assert await calculator.run({"input": "10 / 2"}) == "5.0"

    @pytest.mark.asyncio
    async def test_complex_expression(self, calculator):
        """测试混合运算和优先级"""
        assert await calculator.run({"input": "2 + 3 * 4"}) == "14"
        assert await calculator.run({"input": "(2 + 3) * 4"}) == "20"

    @pytest.mark.asyncio
    async def test_power_and_neg(self, calculator):
        """测试幂运算和负数"""
        assert await calculator.run({"input": "2 ** 3"}) == "8"
        assert await calculator.run({"input": "-5"}) == "-5"

    @pytest.mark.asyncio
    async def test_math_functions(self, calculator):
        """测试内置数学函数"""
        assert await calculator.run({"input": "sqrt(16)"}) == "4.0"
        assert await calculator.run({"input": "abs(-10)"}) == "10"
        assert await calculator.run({"input": "round(3.6)"}) == "4"
        assert await calculator.run({"input": "max(1, 5, 3)"}) == "5"
        assert await calculator.run({"input": "min(1, 5, 3)"}) == "1"

    @pytest.mark.asyncio
    async def test_trigonometry(self, calculator):
        """测试三角函数"""
        result = float(await calculator.run({"input": "sin(pi)"}))
        assert abs(result) < 1e-10
        assert await calculator.run({"input": "cos(0)"}) == "1.0"

    @pytest.mark.asyncio
    async def test_empty_input(self, calculator):
        """测试空输入"""
        with pytest.raises(ToolException):
            await calculator.run({"input": ""})

    @pytest.mark.asyncio
    async def test_missing_input(self, calculator):
        """测试缺少输入参数"""
        with pytest.raises(ToolException):
            await calculator.run({})

    @pytest.mark.asyncio
    async def test_unsupported_function(self, calculator):
        """测试不支持的函数"""
        with pytest.raises(ToolException):
            await calculator.run({"input": "eval('1+1')"})

    @pytest.mark.asyncio
    async def test_invalid_syntax(self, calculator):
        """测试非法语法"""
        with pytest.raises(ToolException):
            await calculator.run({"input": "2 + +"})

    @pytest.mark.asyncio
    async def test_expression_parameter(self, calculator):
        """测试 'expression' 参数作为 'input' 的别名"""
        result = await calculator.run({"expression": "10 + 10"})
        assert result == "20"

    def test_get_parameters(self, calculator):
        """测试获取参数定义"""
        params = calculator.get_parameters()
        assert len(params) == 1
        assert params[0].name == "expression"
        assert params[0].required is True
