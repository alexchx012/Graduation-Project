前情提要：

isaacsim已安装，C:\isaacsim，自带python解释器

isaaclab已安装，D:\IsaacLab，自带conda环境=自带python解释器

robot_lab安装路径：D:\isaaclab-kuozhan\robot_lab



![image-20260125221628692](D:\Graduation-Project\docs\problems\image-20260125221628692.png)

gpt建议安装完成之后执行`python -c "import robot_lab; print(robot_lab.__file__)"`

安装完成之后在lab的conda环境中尝试运行，在lab和robotlab根目录下均运行失败，报错缺失omni模块

折腾后发现如果只是用isaacsim的python运行检查语句也提示缺失，是因为这个kit是依赖于SimulationApp的，必须要先在在conda环境的脚本里创建SimulationApp，这会触发注入与加载程序，然后再import robot_lab，就可以了。而且要通过isaaclab根目录的isaaclab.bat启动脚本运行py文件才行。主要是因为robot_lab顶层 import 会触发 Isaac Lab/omni 相关导入链路，不初始化 Kit 就会报错。

之后要是有任何需要注入omni或者isaaclab环境的脚本，统一用isaaclab.bat -p执行

如果自己写脚本时必须在导入`omni.*`依赖或者lsaaclab env前创建一个SimulationApp实例

```python
from omni.isaac.kit import SimulationApp
app = SimulationApp({"headless": True})

# 这里再 import 依赖 omni 的东西
import robot_lab
# import isaaclab / isaaclab_tasks / omni.timeline 等

# ...逻辑...

app.close()

```

