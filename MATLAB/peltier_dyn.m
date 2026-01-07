function [T1_dot, T2_dot] = fcn(T1, T2, I_total)
    % Constants (Define these or pass them in)
    n = 2.3279; C = 4.9013; K = 0.001255; G = 0.01279;
    R = 2.0000; alpha = 0.065036; Tinf = 24.3400;

    dT = T2 - T1;
    
    T1_dot = (1/C) * ( K*sign(dT)*abs(dT)^n + 0.5*R*I_total^2 + alpha*I_total*T1 + G*(Tinf - T1));
    T2_dot = (1/C) * (-K*sign(dT)*abs(dT)^n + 0.5*R*I_total^2 - alpha*I_total*T2 + G*(Tinf - T2));
end
