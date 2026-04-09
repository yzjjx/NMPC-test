import casadi.*

x1 = MX.sym('x1');
x2 = MX.sym('x2');
x = [x1;x2];
u = MX.sym('u');

ode = [(1-x2^2)*x1-x2+u;x1];
size(ode)

f = Function('f',{x,u},{ode});
f([0.2;0.8],0.1)

%% 离散化
T = 10;%总控制时常
N = 20;% 控制区间，即计算机0.5s改变一次控制量

% 配置积分器选项，Integrator Options
intg_options = struct;
intg_options.tf = T/N;
intg_options.simplify = true;
intg_options.number_of_finite_elements = 4;%切分控制步

dae = struct;
dae.x = x;
dae.p = u;
dae.ode = f(x,u);
% 龙格库塔数值积分
intg = integrator('intg','rk',dae,intg_options)

%% 
%给定初始状态0，1，控制参数为0
% intg([0;1],0,[],[],[],[])
res = intg('x0',x,'p',u);
% 会返回一个结果结构体 res (result)。里面包含了积分器算出来的所有结果。
x_next = res.xf
F = Function('F',{x,u},{x_next},{'x','u'},{'x_next'})
F([0;1],0)
F([0.1;0.9],0.1)

%%
F
sim = F.mapaccum(N)

x0 = [0;1];
res = sim(x0,cos(1:N));

figure
tgrid = linspace(0,T,N+1);
plot(tgrid,full([x0 res]));
legend('x1','x2');
xlabel('t[s]');
%% 引入未知量
U = MX.sym('U',1,N);
X_all = sim(x0, U);
X1 = X_all(1, :);
J = jacobian(X1,U);
size(J);
spy(J);

Jf = Function('F',{U},{J});
imshow(full(Jf(0)));

%% 最优化
opti = casadi.Opti();

x = opti.variable(2,N+1);
u = opti.variable(1,N);
p = opti.parameter(2,1);

opti.minimize(sumsqr(x)+sumsqr(u));

for k=1:N
    opti.subject_to(x(:,k+1)==F(x(:,k),u(:,k)));
end
opti.subject_to(-1<=u<=1);
opti.subject_to(x(:,1)==p);

opti

%%
% 第一步：配置求解器算法 (注意这里是 solver)
opti.solver('sqpmethod', struct('qpsol','qrqp'));

% 第二步：给参数 p 赋初值 (注意这里用分号 ; 变成列向量)
opti.set_value(p, [0; 1]);

% 第三步：正式点火求解 (solve 里面什么都不用填)
sol = opti.solve();
%% 
figure
hold on
plot(tgrid,sol.value(x));
stairs(tgrid,[sol.value(u) nan],'-.');
xlabel('t[s]');
ylabel('Values');  
legend('x1', 'x2', 'u');   
% nice_fig

spy(jacobian(opti.g,opti.x));

spy(hessian(opti.f,opti.x));

opts = struct;
opts.qpsol = 'qrqp';
opts.print_header = false;
opts.print_iteration = false;
opts.print_time = false;
opts.qpsol_options.print_iter = false;
opts.qpsol_options.print_header = false;
opts.qpsol_options.print_info = false;
opti.solver('sqpmethod',opts);

%%
M = opti.to_function('M',{p},{u(:,1)},{'p'},{'u_opt'})

%% MPC loop
X_log = [];
U_lod = [];

x = [0,1];
for i = 1:4*N
    u = full(M(x));

    U_log(:,i) = u;
    X_log(:,i) = x;

    x = full(F(x,u))+[0;rand*0.02];
end

%% 
figure
hold on
tgrid_mpc = linspace(0,4*T,4*N+1);
plot(tgrid_mpc,[x0 X_log]);
stairs(tgrid_mpc,[U_log nan],'-.')
xlabel('t[s]');
legend('x1','x2','u')