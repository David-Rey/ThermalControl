
clear; clc; close all;

%{
Polytropic Index (n):         2.3279
Thermal Capacitance (C):      4.9013 J/K
Thermal Conductance (K):      0.001255 W/K
Ambient Conductance (G):      0.01279 W/K
Electrical Resistance (R):    2.0000 Ω
Seebeck Coefficient (alpha):  -0.065036 V/K
Ambient Temperature (Tinf):    24.3400 C
%}

n = 2.3279;
C = 4.9013;
K = 0.001255;
G = 0.01279;
R = 2.0000;
alpha = 0.065036;
Tinf = 24.3400;

I_val = 0.090303;


system = @(v) [ ...
	(1/C) * ( K*sign(v(2)-v(1))*abs(v(2)-v(1))^n + 0.5*R*I_val^2 + alpha*I_val*v(1) + G*(Tinf - v(1))); ...
    (1/C) * (-K*sign(v(2)-v(1))*abs(v(2)-v(1))^n + 0.5*R*I_val^2 - alpha*I_val*v(2) + G*(Tinf - v(2))) ...
];


T_guess = [1000, 1000];

% Solve numerically
options = optimoptions('fsolve', 'Display', 'iter');
T_solution = fsolve(system, T_guess, options);

fprintf('Solution: T1 = %.4f, T2 = %.4f\n', T_solution(1), T_solution(2));

syms T1 T2 I

T1_dot = (1/C) * ( K*sign(T2-T1)*abs(T2-T1)^n + 0.5*R*I^2 + alpha*I*T1 + G*(Tinf - T1));
T2_dot = (1/C) * ( -K*sign(T2-T1)*abs(T2-T1)^n + 0.5*R*I^2 + alpha*I*T2 + G*(Tinf - T2));

T = [T1 T2];

A_sym = jacobian([T1_dot, T2_dot], T);
B_sym = jacobian([T1_dot, T2_dot], I);

A_num = matlabFunction(A_sym, 'Vars', {[T1, T2], I});
B_num = matlabFunction(B_sym, 'Vars', {[T1, T2], I});

numerical_A = A_num(T_solution, I_val);
numerical_B = B_num(T_solution, I_val);

C = [0 1];
D = [0];
[a, b] = ss2tf(numerical_A, numerical_B, C, D);

Gp = tf(a,b);
Gi = tf([1], [0.5 1]);  % Inner loop responce
H = tf([1], [5 1]);

G_to_tune = Gp * Gi * H;
opts = pidtuneOptions('PhaseMargin', 60);
pid_result = pidtune(G_to_tune, 'PDF', opts);
C_tf = tf(pid_result);

sys_cl = feedback(C_tf * Gi * Gp, H);

step(sys_cl)
%margin(sys_cl)
grid on

%rlocus(sys_cl)
%grid on

%sys_cl = feedback(G, 1);
%CL_poles = pole(sys_cl);
%OL_poles = eig(numerical_A);

%disp('Open-Loop (Plant) Poles:'); disp(OL_poles);
%disp('Closed-Loop (Controlled) Poles:'); disp(CL_poles);

%rlocus(sys_cl)
%grid on


