clear 
clc
close all

ncdisp("4_cylinders_square.nc");
added_mass=ncread("4_cylinders_square.nc","added_mass");
radiation_damping=ncread("4_cylinders_square.nc","radiation_damping");
omega=ncread("4_cylinders_square.nc","omega");
M=ncread("4_cylinders_square.nc","inertia_matrix");
hydrostatic_stiff=ncread("4_cylinders_square.nc","hydrostatic_stiffness");
F_diff=ncread("4_cylinders_square.nc","diffraction_force");
F_froude=ncread("4_cylinders_square.nc","Froude_Krylov_force");
M_total=zeros(4,4,100);
F_diff_combined=zeros(4,100);
F_froude_combined=zeros(4,100);
B_pto=29000; %Ns/m
C=zeros(4,4,100);
C_inverse=zeros(4,4,100);
C_original=zeros(4,4,100);
C_inverse_original=zeros(4,4,100);
RAO=zeros(4,100);
RAO_original=zeros(4,100);


% merge real and imaginary part of diffraction force and Froude Krylov,
% merge mass matrix and added mass matrix
% setup C_ij 
for i=1:100
    for j=1:4
        F_diff_combined(j,i)=F_diff(j,1,i,1)+1i*F_diff(j,1,i,2);
        F_froude_combined(j,i)=F_froude(j,1,i,1)+1i*F_froude(j,1,i,2);
    end
    M_total(:,:,i)=M+added_mass(:,:,i);
    C(:,:,i)=-(omega(i))^2*(M_total(:,:,i))+1i*omega(i)*(radiation_damping(:,:,i)+B_pto)+hydrostatic_stiff(:,:);
    C_inverse(:,:,i)=inv(C(:,:,i));
    C_original(:,:,i)=-(omega(i))^2*(M_total(:,:,i))+1i*omega(i)*(radiation_damping(:,:,i))+hydrostatic_stiff(:,:);
    C_inverse_original(:,:,i)=inv(C_original(:,:,i));
end
F_excitation=F_froude_combined+F_diff_combined;

% calculate RAO
for i=1:100
    RAO(:,i)=C_inverse(:,:,i)*F_excitation(:,i);
    RAO_original(:,i)=C_inverse_original(:,:,i)*F_excitation(:,i);
end

z_a=zeros(9,15,4);
omega_index=zeros(15,1);
T_2=1:15;
omega_2=2*pi./T_2;
H_sign=1:9;            
z_sign=H_sign/2;                         %significant amplitude
power=zeros(9,15,4);


%find the index of nearest omega to omega_2
for i=1:15
     [~,omega_index(i)]=min(abs(omega-omega_2(i)));
end
% calculate response heave motion
for i=1:9
    for j=1:15
        for k=1:4
            z_a(i,j,k)=z_sign(i)*abs(RAO(k,omega_index(j)));  
        end
    end
end

for i=1:9
    for j=1:15
        for k=1:4
            power(i,j,k)=0.5*(omega_2(j))^2*B_pto*(z_a(i,j,k))^2/1000; %kW
        end
    end
end
power_total=power(:,:,1)+power(:,:,2)+power(:,:,3)+power(:,:,4);

% Plot RAO for each device
for i = 1:4
    figure()
    plot(omega, abs(RAO_original(i,:)))
    title(sprintf('Heave RAO Cylinder %d', i))
    xlabel('Wave frequency (rad/s)')
    ylabel('RAO (-)')
    % saveas(gcf, sprintf('D:/桌面/Figures/RAO_cylinder_%d.png', i))
end


% Plot damped RAO for each device
for i=1:4
    figure()
    plot(omega,abs(RAO(i,:)))
    title(sprintf("Damped Heave RAO Cylinder %d", i))
    xlabel("Wave frequency(rad/s)")
    ylabel("RAO(-)")
    % saveas(gcf, sprintf('D:/桌面/Figures/RAO_damped_cylinder_%d.png', i))
end

% Plot added mass 
figure()
plot(omega, squeeze(added_mass(1, 1, :)))
hold on
plot(omega, squeeze(added_mass(2, 2, :)))
plot(omega, squeeze(added_mass(3, 3, :)))
plot(omega, squeeze(added_mass(4, 4, :)))
hold off
title('Added Mass')
legend('cylinder 1','cylinder 2', 'cylinder 3', 'cylinder 4')
xlabel("Wave frequency(rad/s)")
ylabel("Added mass(-)")

% Plot excitation force
figure()
plot(omega, abs(F_excitation(1,:))/1000)
hold on
plot(omega, abs(F_excitation(2,:))/1000)
plot(omega, abs(F_excitation(3,:))/1000)
plot(omega, abs(F_excitation(4,:))/1000)
hold off
title('Excitation Force')
legend('cylinder 1','cylinder 2', 'cylinder 3', 'cylinder 4')
xlabel("Wave frequency(rad/s)")
ylabel("Forcek(N)")



figure()
imagesc(power_total);
colormap default
caxis([min(power_total(:)) max(power_total(:))])
textStrings = num2str(power_total(:),'%0.1f');          % Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));    % Remove any space padding
[x,y] = meshgrid(1:15,1:9);                          % Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),'HorizontalAlignment','center','FontSize',6);    % Plot the strings
midValue = mean(get(gca,'CLim'));               % Get the middle value of the color range
textColors = repmat(power_total(:) < midValue,1,3);     % Choose white or black for the text color
set(hStrings,{'Color'},num2cell(textColors,2)); % Change the text colors
colorbar
set(gca,'XTick',1:15,'YTick',1:9)
xlabel('Wave Period [s]')
ylabel('Significant Wave Height [m]')
ylabel(colorbar,'Power(kW)')
title(['Total Power Matrix for Damping = ' num2str(B_pto(end)) ' [Ns/m]'])
