@echo off
chcp 65001 >nul
echo ========================================
echo 批量运行 find_duals.py 脚本
echo ========================================
echo.

REM 检查Python是否可用
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 找不到Python，请确保Python已安装并添加到PATH
    pause
    exit /b 1
)

REM 检查find_duals.py是否存在
if not exist "find_duals.py" (
    echo 错误: 找不到find_duals.py文件
    echo 请确保您在正确的目录中运行此脚本
    pause
    exit /b 1
)

echo 请选择运行方式:
echo 1. 使用默认次数 (100次)
echo 2. 自定义运行次数
echo 3. 快速测试 (10次)
echo 4. 大量收集 (1000次，用于完整提取)
echo.

set /p choice="请输入选择 (1-4): "

if "%choice%"=="1" (
    set iterations=100
) else if "%choice%"=="2" (
    set /p iterations="请输入运行次数: "
) else if "%choice%"=="3" (
    set iterations=10
) else if "%choice%"=="4" (
    set iterations=1000
) else (
    echo 无效选择，使用默认值100次
    set iterations=100
)

echo.
echo 将运行 %iterations% 次，预计生成 %iterations%0,000 个对偶点
echo 按任意键开始，或Ctrl+C取消...
pause >nul

python batch_find_dual_points.py %iterations%

echo.
echo 批量运行完成!
pause 