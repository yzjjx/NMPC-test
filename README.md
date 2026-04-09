NMPC Test

首先需要利用CasADi来构建计算图（computationla graphs），以此来定义和计算常微分方程（ODE）

[CasADi - Gets](https://web.casadi.org/get/)

范德波尔振荡器：

![image-20260409200420812](C:\Users\GALAXY\AppData\Roaming\Typora\typora-user-images\image-20260409200420812.png)

给出常微分方程通用数学表达式：

***********

step1：computational graphs

用于计算下一步的x1和x2的状态，建立非线性系统模型，其中`MX.`是CasADi中用来声明符号变量的命令，最后的f=Function是将微分方程封装成了一个可以直接调用的函数，函数的名字为f。

```matlab
import casadi.*

x1 = MX.sym('x1');
x2 = MX.sym('x2');
x = [x1;x2];
u = MX.sym('u');

ode = [(1-x2^2)*x1-x2+u;x1];
size(ode)

f = Function('f',{x,u},{ode});
f([0.2;0.8],0.1)
```

step2：time-integration methods

对微分方程进行离散化，配置数值积分器，预测范围为T = 10s，分为N=20区间，控制周期为T = T/N =0.5s；

```matlab
T = 10;% 总控制时常
N = 20;% 控制区间，即计算机0.5s改变一次控制量

% 配置积分器选项，Integrator Options
intg_options = struct;
intg_options.tf = T/N;
intg_options.simplify = true;
intg_options.number_of_finite_elements = 4;% 切分控制步

dae = struct;
dae.x = x;
dae.p = u;
dae.ode = f(x,u);
% 龙格库塔数值积分
intg = integrator('intg','rk',dae,intg_options)
```

step3：生成离散状态转移函数

```matlab
res = intg('x0',x,'p',u);
% 会返回一个结果结构体 res (result)。里面包含了积分器算出来的所有结果。
x_next = res.xf
F = Function('F',{x,u},{x_next},{'x','u'},{'x_next'})
```

step4：开环前向仿真

```matlab
sim = F.mapaccum(N)
x0 = [0;1];
res = sim(x0,cos(1:N));
```

step5：雅可比矩阵分析

画出雅可比矩阵的稀疏结构图

```matlab
U = MX.sym('U',1,N);
X_all = sim(x0, U);
X1 = X_all(1, :);
J = jacobian(X1,U);
size(J);
spy(J);
```

step6：建立最优化问题



********

MATLAB导入外部包

![image-20260409201407854](C:\Users\GALAXY\AppData\Roaming\Typora\typora-user-images\image-20260409201407854.png)

下载文件并且解压

![image-20260409201455143](C:\Users\GALAXY\AppData\Roaming\Typora\typora-user-images\image-20260409201455143.png)

添加到matlab路径

![image-20260409201716187](C:\Users\GALAXY\AppData\Roaming\Typora\typora-user-images\image-20260409201716187.png)

![image-20260409201731200](C:\Users\GALAXY\AppData\Roaming\Typora\typora-user-images\image-20260409201731200.png)

