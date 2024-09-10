clc;clear;close all

%% Information, Copyright and Contact Details

% GENERAL INFORMATION

% This code simulates a network of spiking neural network. The neuron
% models are the "FitzHugh-Nagumo" and the synaptic connections are
% "diffusive". 


% CORRESPONDING PAPER AND CITATION INFORMATION

% The following code is added as a supplementary file to the paper titled
% "An Operator-Theoretic Framework to Simulate Neuromorphic Circuits",
% published in " 2024 Conference on Decision and Control (CDC)" by "Amir
% Shahhosseini, Thomas Chaffey and Rodolphe Sepulchre". All rights are
% reserved.


% CODE DETAILS AND CONTACT INFORMATION

% Date: Sep. 10, 2024
% Coded By: Amir Shahhosseini
% Affiliation: KU Leuven
% Contact Information: amir.shahhosseini@kuleuven.be
% All rights reserved.

%% MAINTENANCE AND LIMITATIONS

% Please bear the following points when modifying this code as neglecting
% these issues can break the code.

% 1. Prior knowledge of the period: The term "T_period" that describes the
% period of the oscillations for the network must be known prior to the
% simulation of the network. This is due to the fact that the simulation is
% performed on the space of periodic square-integrable signals and
% periodicity is a requirement. There are theoretical methods to estimate
% the period of a single spiking neurons but the most practical and
% computationally cheap way is just to simulate one FN neuron using
% numerical integration method and use that information for this purpose.
% The requirement of periodicity is relaxed in our next publication, 
% "Modeling and Simulation of Large-Scale Neuromorphic Networks", IEEE
% Transcations of Automatic Control (In preparation).

% WARNING: Changing the spiking neuron's variables (L_mean, C_Mean, R_mean)
% affects the period of the spiking neuron and any adjustments of these
% parameters must be followed by altering the T_period. Due to this exact
% reason, having a high level of deviation from mean values results in
% inaccurate simulations. 

% 2. The diffusive coupling of this network CANNOT be set to be zero. In
% order to eliminate a synaptic connection between two neurons, make the
% resistance of the corresponding synaptic connection very large
% (R_syn(i,j) and R_syn(j,i)). This corresponds to no synaptic connections.

% WARNING: Setting the synaptic resistance to zero breaks the code for
% obvious reasons. Refer to operator M_2 of the paper for elaboration.

% 3. This code is designed to simulate networks and the current version is
% incapable of treating a single neuron. setting the number of neurons (m)
% to 1 breaks the code. In case of a dire need to simulate a single
% isolated neuron, simulate the network with two neurons and make the
% resistance of their coupling very high.

% 4. The initial conditions of this network are set arbitrarily (harmonics)
% but it is possible to accelerate this simulation framework by orders of
% magnitudes upon using better initial conditions. This can be done by
% changing lines 167 and 171.

%% System Parameters

T_period = 55.54;            % Corresponding to L_mean = 20, C_mean = 1 and R_mean = 1;

m = 100;                     % Number of Neurons

L_mean = 20;                 % Mean value of the inductance of the FN circuit
C_mean = 1;                  % Mean value of the capacitance of the FN circuit
R_mean = 1;                  % Mean value of the resistance of the FN circuit

deviation = 0.2;             % Maximum deviation of mean values from nominal values

% Inductor's Inductance

L = L_mean + deviation*L_mean*(rand(1,m)-0.5);

% Capacitor's capacitance

C = C_mean + deviation*C_mean*(rand(1,m)-0.5);

% Resistance of the Inductive Branch

R = R_mean + deviation*R_mean*(rand(1,m)-0.5);

% Diffusive Coupling

R_syn_mean = 5;             % Mean value of the synaptic connection's resistance
R_syn = zeros(m);
for i = 2:m
    for j = 1:i-1
        R_syn(i,j) = R_syn_mean + deviation*R_syn_mean*(rand()-0.5);
    end
end

R_syn = R_syn + transpose(R_syn);

Inv_R_syn = 1./(R_syn + eye(m)) - eye(m);
temp_Inv_R_syn = Inv_R_syn + eye(length(Inv_R_syn));

%% FFT Properties

Fs = 10;                  % Sampling frequency                    
T = 1/Fs;                 % Sampling period       
N = ceil(T_period/T);     % Length of signal
t = (0:N-1)*T;            % Time vector  - From 0 to 1 sec with step size 0.01 sec

%% Solver parameters and Operators

alpha = 0.7;                % coefficient of the resolvent corresponding to Eq. 5

a  = zeros(1,N);

for k = 0:N-1
    a(k+1) = exp(-((2*1i*pi)/N)*k);
end
D = (1/T)*(eye(N)-diag(a));
d = diag(D);

S1 = zeros(1,m*N);
S4 = zeros(1,m*N);

for j = 0:m-1
    S1((N*j+1):(j+1)*N) = C(j+1).*d;
    S4((N*j+1):(j+1)*N) = L(j+1).*d;
end

% Using the Idea Both Schur's complement and the fact that the blocks are
% diagonal!

block11 = ones(1,m*N) + alpha*S1;
block12 = alpha*ones(1,m*N);
block21 = -alpha*ones(1,m*N);
block22 = ones(1,m*N) + alpha*S4;

inv_block11 = 1./block11;
inv_block12 = 1./block12;
inv_block21 = 1./block21;
inv_block22 = 1./block22;

TEMP = 1./(block22 - block21.*inv_block11.*block12);

B11 = inv_block11 + inv_block11.*block12.*TEMP.*block21.*inv_block11;
B12 = -inv_block11.*block12.*TEMP;
B21 = -TEMP.*block21.*inv_block11;
B22 = TEMP;


%%%%%%%%%%% Pre-allocation of the solution/auxilary vectors %%%%%%%%%%

max_iteration = 130;                      % Maximum Number of Iteration Allowed

x = zeros(max_iteration,m*N);
z = zeros(max_iteration,m*N);
Z = zeros(1,m*N);

for j = 0:m-1
    z(1,(N*j+1):(j+1)*N) = cos(((2*pi)/T_period)*t);         % Pre-allocation of the initial conditions
end

for j = m:2*m-1
    z(1,(N*j+1):(j+1)*N) = sin(((2*pi)/T_period)*t);         % Pre-allocation of the initial conditions
end

%% Difference of Monotone Douglas-Rachford Splitting Algorithm


threshold = 1e-2;
coeff = 0.7;                                   % Coefficient for the Guarded-Newton's Method
inv_alpha = (1/alpha);

tic                                            % To time the algorithm
for i = 1:max_iteration

    for j = 0:2*m-1
        Z((N*j+1):(j+1)*N) = fft(z(i,(N*j+1):(j+1)*N));
    end

    Y1 = B11.*Z(1:length(B11)) + B12.*Z(length(B11)+1:end);
    Y2 = B21.*Z(1:length(B21)) + B22.*Z(length(B21)+1:end);

    Y = [Y1 Y2];

    for j = 0:2*m-1
         x(i+1,(N*j+1):(j+1)*N) = real(ifft(Y((N*j+1):(j+1)*N)));
    end

    B_s = transpose(reshape(x(i+1,1:m*N),[N,m]));

    for io = 0:m-1
        S11 = sum(transpose(temp_Inv_R_syn(io+1,1:m)).*B_s);
        tempp(io*N+1:(io+1)*N) = S11;
    end

    FAST_COMP = [tempp zeros(1,m*N)];

    temp = 2*x(i+1,:) - z(i,:) + (alpha*FAST_COMP);

    temp2 = zeros(1,m*N);

    for o = 0:m-1
        Rc = sum(Inv_R_syn(o+1,:));

        for j = 1:N

            iteration = 1;
            error_algo = 1;
            l = -10;
            u = +10;

            if j == 1

                x_start = 1;
                v = temp(N*o+j);
                while error_algo > threshold
        
                    if iteration == 1
                        x_local = x_start;
                    else
                        x_local = x_local - g/d2phi;
                    end
            
                    low = 0.5*(l+u) - coeff*(u-l)*0.5;
                    high = 0.5*(u+l) + coeff*(u-l)*0.5;
            
                    if x_local > high
                        x_local = high;
                    elseif x_local < low
                        x_local = low;
                    end
            
                    h = (x_local^3)/3 + x_local*Rc;              % gradient f
                    g = h + (1/alpha)*(x_local-v);               % gradient phi
                    d2phi = (x_local^2) + (1/alpha) + Rc;        % second gradient of phi
            
                    if g > 0
                        l = max(l,x_local-alpha*g);
                        u = min(u,x_local);
                    else
                        l = max(l,x_local);
                        u = min(u,x_local-alpha*g);
                    end
            
                    error_algo = u - l;
                    iteration = iteration + 1;
                end

                temp2(N*o+j) = x_local;

            else
                x_start = temp2(N*o+j-1);
                v = temp(N*o+j);

                while error_algo > threshold
        
                    if iteration == 1
                        x_local = x_start;
                    else
                        x_local = x_local - g/d2phi;
                    end
            
                    low = 0.5*(l+u) - coeff*(u-l)*0.5;
                    high = 0.5*(u+l) + coeff*(u-l)*0.5;
            
                    if x_local > high
                        x_local = high;
                    elseif x_local < low
                        x_local = low;
                    end
            
                    h = x_local*(Rc + (x_local*x_local)/3);      % gradient f
                    g = h + inv_alpha*(x_local-v);               % gradient phi
                    d2phi = (x_local^2) + inv_alpha + Rc;        % second gradient of phi
            
                    if g > 0
                        l = max(l,x_local-alpha*g);
                        u = min(u,x_local);
                    else
                        l = max(l,x_local);
                        u = min(u,x_local-alpha*g);
                    end
            
                    error_algo = u - l;
                    iteration = iteration + 1;
                end
                temp2(N*o+j) = x_local;
            end
        end
    end


    for j = m:2*m-1
        temp2((N*j+1):(j+1)*N) = (1/(1+R(j-m+1)*alpha)).*temp((N*j+1):(j+1)*N);
    end

    z(i+1,:) = z(i,:) - x(i+1,:) + temp2;

    % Graphical Depiction of the iterations: this will depend on the value
    % of alpha (line 119) and also how frequently you look at the captures.
    % This is currently disabled because this grapical interface slows the
    % simulation.

    % if mod(i,5) == 1
    %     plot(t,x(i,1:N))
    %     pause(0.1)
    % end

end
toc


% Plotting the voltage of the first neuron at the final iteration.
for j = 0
    plot(t,x(i,j*N+1:(j+1)*N),'LineWidth',3)
    hold on
end