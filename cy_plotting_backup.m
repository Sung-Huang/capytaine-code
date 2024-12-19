clear 
clc
close all
format long

ncdisp("cy_rao.nc")
rao_real=ncread("cy_rao.nc","real"); % rao exported from capytaine
rao_imag=ncread("cy_rao.nc","imag");
rao=rao_real+rao_imag*1i;
s=rao(3,:);% heave response amplitude operator

ncdisp("cy_dataset.nc");
added_mass=ncread("cy_dataset.nc","added_mass");
radiation_damping=ncread("cy_dataset.nc","radiation_damping");
omega=ncread("cy_dataset.nc","omega");
M=ncread("cy_dataset.nc","inertia_matrix");
hydrostatic_stiff=ncread("cy_dataset.nc","hydrostatic_stiffness");
A=squeeze(added_mass(3,3,:)); %Added mass matrix
B=squeeze(radiation_damping(3,3,:));
K=hydrostatic_stiff(3,3); %Heave hydrostatic stiffness
F_diff=ncread("cy_dataset.nc","diffraction_force");
F_froude=ncread("cy_dataset.nc","Froude_Krylov_force");
F_d_heave=zeros(100,1);
F_f_heave=zeros(100,1);
M_total=zeros(100,1);
C=zeros(100,1);
%% Calculate RAO
% merge real and imaginary part of diffraction force and Froude Krylov,
% merge mass matrix and added mass matrix
% setup C_ij 
for i=1:100
    F_d_heave(i)=F_diff(3,1,i,1)+1i*F_diff(3,1,i,2);
    F_f_heave(i)=F_froude(3,1,i,1)+1i*F_froude(3,1,i,2);
    M_total(i)=M(3,3)+added_mass(3,3,i);
    C(i)=-(omega(i))^2*(M_total(i))+1i*omega(i)*radiation_damping(3,3,i)+hydrostatic_stiff(3,3);
end
F_excitation=F_d_heave+F_f_heave;
% calculate RAO
RAO=zeros(100,1);
for i=1:100
    RAO(i)=F_excitation(i)./C(i);
end



%% Find natural frequency
K=hydrostatic_stiff(3,3); %Heave hydrostatic stiffness
A=squeeze(added_mass(3,3,:)); %Added mass matrix

value=zeros(100,1);

T_sign=sort(2*pi./omega);             %averaged zero-crossing period
H_sign=1:9;            
z_sign=H_sign/2;                         %significant amplitude
power=zeros(9,15);

% Find the intercept point which is the natural frquency
natural_frquency_index=zeros(100,1);
for i=1:100
    value(i)=(omega(i))^2-(K/(A(i)+M(3,3)));
end

% Plot the (omega,value) to find the intersection=natrural frequency
figure()
hold on
plot(omega,value)
yline(0,'-','Zero Line');
plot([1.2209, 1.2209],[-20. 0])
xlabel('Wave frequency(rad/s)')
ylabel('Value')
title('Natural Frequency Derivation')
hold off

% Find the closest value to zero
for i=1:100
    [~,natural_frquency_index(i)]=min(abs(value-0));
end
natural_frequency=omega(natural_frquency_index(1)); 
fprintf('Natural frequency of the device is: %.5f\n', natural_frequency)

%% Find optimal B_pto
R_1=zeros(100,1);
X_1=zeros(100,1);
b_pto=zeros(100,1);
zeta=zeros(100,1); %percentage of critical damping
y=zeros(100,300);
optimal_b_test=zeros(21,1);
y0=1;
omega_n=natural_frequency;
wave_scatter_T=linspace(4.25,14.25,21);
wave_scatter_omega=2*pi./wave_scatter_T;

for i=1:length(wave_scatter_omega)
    optimal_b_test(i)=calculate_optimal_Bpto(wave_scatter_omega(i),natural_frquency_index);
end
optimal_b=calculate_optimal_Bpto(omega_n,natural_frquency_index);

figure()
plot(wave_scatter_T, optimal_b_test)
xlabel('Wave period [s]')
ylabel('Optimal PTO damping coefficient [Ns/m]')
title('Optimal B_{PTO} under different wave period')

%% Calculate power matrix
% Value for power matrix
B_pto=29000; %Ns/m
z_a=zeros(9,15);
omega_index=zeros(15,1);
T_2=1:15;
omega_2=2*pi./T_2;

% calculate RAO include b_pto
C_b=zeros(100,1);
for i=1:100
    C_b(i)=-(omega(i))^2*(M_total(i))+1i*omega(i)*(radiation_damping(3,3,i)+B_pto)+hydrostatic_stiff(3,3);
end
RAO_b=zeros(100,1);
for i=1:100
    RAO_b(i)=F_excitation(i)./C_b(i);
end

%find the index of nearest omega to omega_2
for i=1:15
     [~,omega_index(i)]=min(abs(omega-omega_2(i)));
end
for i=1:9
    for j=1:15
        z_a(i,j)=z_sign(i)*abs(RAO_b(omega_index(j)));    
    end
end

% calculate power
for i=1:9
    for j=1:15
        power(i,j)=0.5*(omega_2(j))^2*B_pto*(z_a(i,j))^2/1000; %kW
    end
end


figure()
plot(omega,squeeze(added_mass(3,3,:)))
xlabel("wave frequency(rad/s)")
ylabel("added mass")
title("Heave added mass")
figure()
plot(omega,squeeze(radiation_damping(3,3,:)))
xlabel("wave frequency(rad/s)")
ylabel("radiation damping")
title ("Heave radiation damping")


figure()
plot(omega,abs(rao(3,:)))
title("RAO imported from Capytaine")
xlabel("wave frequency(rad/s)")
ylabel("Heave RAO(-)")

%Compare figure 4 with figure 5
figure()
plot(omega,abs(RAO))
title("RAO (Manually calculation)")
xlabel("Wave Frequency(rad/s)")
ylabel("Heave RAO(-)")

figure()
plot(omega,abs(RAO_b))
title("RAO (Manually calculation with PTO damping)")

% Heave excitation force
% figure()
% plot(omega, F_excitation)

figure()
imagesc(power);
colormap default
caxis([min(power(:)) max(power(:))])
textStrings = num2str(power(:),'%0.1f');          % Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));    % Remove any space padding
[x,y] = meshgrid(1:15,1:9);                          % Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),'HorizontalAlignment','center','FontSize',6);    % Plot the strings
midValue = mean(get(gca,'CLim'));               % Get the middle value of the color range
textColors = repmat(power(:) < midValue,1,3);     % Choose white or black for the text color
set(hStrings,{'Color'},num2cell(textColors,2)); % Change the text colors
colorbar
set(gca,'XTick',1:15,'YTick',1:9)
xlabel('Wave Period [s]')
ylabel('Significant Wave Height [m]')
ylabel(colorbar,'Power(kW)')
title(['Power Matrix for Damping = ' num2str(B_pto(end)) ' [N/m/s](single device)'])

function optimal_b=calculate_optimal_Bpto(omega_n,natural_frquency_index)
    ncdisp("cy_dataset.nc");
    added_mass=ncread("cy_dataset.nc","added_mass");
    radiation_damping=ncread("cy_dataset.nc","radiation_damping");
    omega=ncread("cy_dataset.nc","omega");
    M=ncread("cy_dataset.nc","inertia_matrix");
    hydrostatic_stiff=ncread("cy_dataset.nc","hydrostatic_stiffness");
    A=squeeze(added_mass(3,3,:)); %Added mass matrix
    B=squeeze(radiation_damping(3,3,:));
    K=hydrostatic_stiff(3,3); %Heave hydrostatic stiffness
    y0=1;
    
    for i=1:100
    
    zeta(i,1)=(B(i,1)/(2*sqrt(K*(M(3,3)+A(i,1)))));
    X_1(i)=1i*omega(i)*(M(3,3)+added_mass(3,3,i))+(hydrostatic_stiff(3,3)/1i);
    
    for j=1:300
    y(i,j)=y0*exp(-omega_n*zeta(i)*j)*cos(sqrt(1-(zeta(i))^2)*omega_n*j);       %free decay motion from t=1 to t=300s
    end
    
    end
    
    y_a=y(13,:); %natural frequency index=13 
    TF=islocalmax(y_a); %find index of x_k
    x_k=y_a(TF); %x_k is the amplitude of the kth oscillation cycle
    dimension=size(x_k);
    number_x_k=dimension(2);
    eqn_y=zeros(number_x_k-2,1);
    
    for i=2:number_x_k-1
    eqn_y(i-1)=(1/2/pi)*(log(x_k(i-1)/x_k(i+1)));
    end
    x_k1=zeros(number_x_k-2,1);
    for i=1:number_x_k-2
        x_k1(i)=x_k(i+1);
    end
    
    c=polyfit(x_k1,eqn_y,1);
    B2=c(1)*(M(3,3)+A(13))*3*pi/4;
    
    for i=1:100
        R_1(i)=radiation_damping(3,3,i)+B2;                                       %R(w)=b_a(w)+R_f
        b_pto(i)=((R_1(i))^2+(imag(X_1(i)))^2)^(1/2);
    end
    
    optimal_b=b_pto(natural_frquency_index(1));
    
    fprintf('Optimal power takeoff coefficient of the device is:%d[Ns/m]', optimal_b)
end
