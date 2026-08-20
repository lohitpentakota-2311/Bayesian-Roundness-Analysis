% % % 24/02
% % Rough estimations
% % (GB coeff, LP Estimations)

% % % % % % signal missing line 75 (26/04-2025)

function [DataOut, DoET, NcOut] = analyzeNCSignals(FileNameExcel,Tests,ElabPar)

CNSubDir = 'MISURE CN TUTTE\';
ExperimentMeas= ElabPar.MeasFiles;

% Function to analyze NC signals for a specified test index.
% Inputs:
% File name excels
%   testIndex - Index of the test to analyze
%
% Outputs:
%   DataOut   - Cell array containing all analysis results during the process (OW, CW)

% % % % Reads the DoE data:
DataDir= ExperimentMeas;
OptsInp = detectImportOptions([DataDir FileNameExcel],'Sheet','DoE','VariableNamesRange','A3','DataRange','A4'); % ,'DataRange',sprintf('A4:CZ%i',LastDoErow)
OptsInp = setvartype(OptsInp, {'provaEseguita_','NomeFileProfiloIniziale','NomeFileProfiloFinale','fareAnalisiProfilo_Iniziale', ...
    'fareAnalisiProfilo_Finale'},'char');  % or 'datetime' or 'double'
DoET=readtable([DataDir FileNameExcel],OptsInp,'UseExcel',true); % 'UseExcel',true to import result of excel formula

% FIXED TEST DATA:
%reads the fixed paramters from the excel file:
OptsFix = detectImportOptions([DataDir FileNameExcel],'Sheet','FixedData','VariableNamesRange','A3','DataRange','A4'); % ,'DataRange','A4:I4');
FixedT=readtable([DataDir FileNameExcel],OptsFix); % 'Sheet','DoE','VariableNamesRange','A3','DataRange',);

MachProc.ContrW.Diam=FixedT.ControWheelDiam_mm;   % Diameter CW in mm
MachProc.OperW.Diam=FixedT.GrindingWheelDiam_mm;   % Diameter OW in mm
MachProc.WP.Diam=FixedT.WPdiameter_mm;   % Diameter WP in mm
MachProc.OperW.InertComp = 209;   % coefficient to compensate OW inertia on OW torque [rpm/s, at the motor]
MachProc.OperW.TrRatio = 1.754; % operating wheel motor/wheel speed ratio
MachProc.OperW.ContTorque = 235.5;  % continuous torque Nm
MachProc.OperW.SpeedRPM=FixedT.GrindingWheelSpeed_rpm; % Commanded OW Speed in RPM
MachProc.Gamma = FixedT.Gamma_deg; % Blade angle [deg] (30: stable; 20 unstable)
MachProc.wplength = FixedT.WPlength_mm;
MachProc.cuttingspeed = FixedT.CuttingSpeed_m_s;
MachProc.ContrW.TrRatio = 14.33;    % motor vel/ control wheel vel
MachProc.ContrW.DynEff = 0.7;   % dynamic efficiency of the Wormgear transmission (hypothesis)
% the motor torque constant is referred to the RMS current, while the measured current is peak:
% it must be divided by sqrt(2)
MachProc.ContrW.Kt = 1.06; % Mail tagliabue 6/17/22: la costante di coppia del motore conduttrice risulta Kt= 1.06 Nm/Arms
MachProc.ContrW.INom = 80;  % e un 80A nominali per l'azionamento conduttrice
MachProc.X.Kt = 1;
MachProc.X.INom = 20;   % Nominal current
MachProc.X.Lead = 5; % lead [mm] of X axis ball screw
MachProc.X.DynEff = 0.9;   % dynamic efficiency of the ball screw transmission (hypothesis)
MachProc.OperW.EffIdle = 0.255; % operating wheel Effort [?] when rotating idle
MachProc.OperW.EffFluid = 0.3; % rough estimate of operating wheel Effort [?] when getting near the workpiece (80-200um)
MachProc.OperW.MultCoeff = 4278; MachProc.OperW.AddCoeff = -148; % coeff to convert LMDAT signal into [W]

% % % % % % % % % % % % % % % % % % % % % % % % %
DataOut= cell(1,length(Tests));
for IT = (Tests)
    if ~(isempty(DoET.fareAnalisiCN{IT})||strcmpi(DoET.fareAnalisiCN{IT},'no')) && ~(isempty(DoET.fareAnalisiProfilo_Finale{IT})||strcmpi(DoET.fareAnalisiProfilo_Finale{IT} ...
            ,'no')) &&  ~(isempty(DoET.fareAnalisiProfilo_Iniziale{IT})||strcmpi(DoET.fareAnalisiProfilo_Iniziale{IT},'no')) % checks if it is required to analyse the NC signals
        % TestNumber = sscanf(DoET.Id{IT},'Test%i',1);
        fprintf('analysing NC data of test: %s\n',DoET.Id{IT});
        % % % % % % % % % % % % % % % % % % % %
        if isfield(ElabPar,'ISign')
            ElabPar = rmfield(ElabPar,'ISign'); % delete all previous signal definitions
        end
        % signals definition differs in different tests:
        if IT<6
            % test G2-G5:
            ElabPar.Label.XposEnc1= 'POSF'; ElabPar.Label.X1posEnc2= 'POSF'; ElabPar.Label.ContrVel3= 'SPEED'; ElabPar.Label.ContrIQ4= 'IQ'; ElabPar.Label.XIQ5= 'IQ'; ElabPar.Label.OperTorq6= 'LMDAT'; ElabPar.Label.OperVel7= 'SPEED'; ElabPar.Label.Err= 'IQ';
            ElabPar.ISign.XposEnc = 1; ElabPar.ISign.X1posEnc = 2; ElabPar.ISign.ContrVel = 3; ElabPar.ISign.ContrIQ = 4; ElabPar.ISign.XIQ = 5; ElabPar.ISign.OperPow = 6; ElabPar.ISign.OperVel = 7; ElabPar.ISign.Err = 8;
            ElabPar.ISign.Units = {'um','um','1/min','A(p)','A(p)','V(p)','1/min','um'};
        elseif IT<301
            ElabPar.Label.XposEnc1= 'POSF'; ElabPar.Label.X1posEnc2= 'POSF'; ElabPar.Label.ContrVel3= 'SPEED'; ElabPar.Label.ContrIQ4= 'IQ'; ElabPar.Label.XIQ5= 'IQ'; ElabPar.Label.OperTorq6= 'LMDAT'; ElabPar.Label.OperVel7= 'SPEED'; ElabPar.Label.X1IQ8= 'IQ';
            ElabPar.ISign.XposEnc = 1; ElabPar.ISign.X1posEnc = 2; ElabPar.ISign.ContrVel = 3; ElabPar.ISign.ContrIQ = 4; ElabPar.ISign.XIQ = 5; ElabPar.ISign.OperPow = 6; ElabPar.ISign.OperVel = 7; ElabPar.ISign.X1IQ = 8;
            ElabPar.ISign.Units = {'um','um','1/min','A(p)','A(p)','V(p)','1/min','A(p)'};
        elseif IT>400 % (i.e., Intaglio measurements) it is having a velocity measuremnt 8 signal (not current !!!!!!!!!!!!!)
            ElabPar.Label.XposEnc1= 'POSF'; ElabPar.Label.X1posEnc2= 'POSF'; ElabPar.Label.ContrVel3= 'SPEED'; ElabPar.Label.ContrIQ4= 'IQ'; ElabPar.Label.XIQ5= 'IQ'; ElabPar.Label.OperTorq6= 'TCMD'; ElabPar.Label.OperVel7= 'SPEED'; ElabPar.Label.X1IQ8= 'LMDAT';
            ElabPar.ISign.XposEnc = 1; ElabPar.ISign.X1posEnc = 2; ElabPar.ISign.ContrVel = 3; ElabPar.ISign.ContrIQ = 4; ElabPar.ISign.XIQ = 5; ElabPar.ISign.OperTorq = 6; ElabPar.ISign.OperVel = 7; ElabPar.ISign.OperPow = 8;
            ElabPar.ISign.Units = {'um','um','1/min','A(p)','A(p)','%','1/min','V(p)'};
        else % IT>300 (i.e., torque measurements)
            ElabPar.Label.XposEnc1= 'POSF'; ElabPar.Label.X1posEnc2= 'POSF'; ElabPar.Label.ContrVel3= 'SPEED'; ElabPar.Label.ContrIQ4= 'IQ'; ElabPar.Label.XIQ5= 'IQ'; ElabPar.Label.OperTorq6= 'TCMD'; ElabPar.Label.OperVel7= 'SPEED'; ElabPar.Label.X1IQ8= 'IQ';
            ElabPar.ISign.XposEnc = 1; ElabPar.ISign.X1posEnc = 2; ElabPar.ISign.ContrVel = 3; ElabPar.ISign.ContrIQ = 4; ElabPar.ISign.XIQ = 5; ElabPar.ISign.OperTorq = 6; ElabPar.ISign.OperVel = 7; ElabPar.ISign.X1IQ = 8;
            ElabPar.ISign.Units = {'um','um','1/min','A(p)','A(p)','%','1/min','A(p)'};
        end
        FileNameNC = [DataDir CNSubDir DoET.NomeFileCN{IT}];
        % % % [DataOut, DoET] = AnaNPhasesCNFanuc(FileNameNC, ISign, MachProc, ElabPar, DoET,IT)
        [DataOut{IT}, NcDoET] = AnaNPhasesCNFanucv1(FileNameNC, ElabPar.ISign, MachProc, ElabPar, DoET, IT);
        NcOut.CWTanForce(IT)= DoET.NC_s1CWTanForce(IT); NcOut.ForceRatio(IT)= DoET.NC_Ftan_FX_ratio(IT);
        % % % % % % % % managing Power to final torque estimation
        if IT<300
            Pow=6; OperVel= 7;
            OperWheelWatt = MachProc.OperW.MultCoeff*DataOut{1, IT}.Data(:,Pow)+MachProc.OperW.AddCoeff; % converted in W and into Nm
            OWtorque = OperWheelWatt./(DataOut{1, IT}.Data(:,OperVel)*2*pi/60);  % torque [Nm]
            % DataOut{1, IT}.Data(:,Pow) = filtfilt(ones(ElabPar.NPSmoothFilt2,1),ElabPar.NPSmoothFilt2,OWtorque);
            DataOut{1, IT}.Data(:,Pow) = OWtorque;
        else
            TMCD=6;
            OWtorque = DataOut{1, IT}.Data(:,TMCD)/100*MachProc.OperW.ContTorque*MachProc.OperW.TrRatio; % torque [Nm] at the operating wheel
            % DataOut{1, IT}.Data(:,TMCD)= filtfilt(ones(ElabPar.NPSmoothFilt2,1),ElabPar.NPSmoothFilt2,OWtorque);
            DataOut{1, IT}.Data(:,TMCD)=OWtorque;
        end
        % % % Fltering on OW acceleration and X velocity added for Non
        % linear feature generation to reduce the noise level in the x axis
        % signals
        % ElabPar.ISign.XposEnc=1; ElabPar.ISign.OperVel=7;
        % VelFilt = [ones(ElabPar.NPVelFilt,1); -ones(ElabPar.NPVelFilt,1)]/ElabPar.NPVelFilt;
        % DT = DataOut{1, IT}.Time(2);
        % XVel = filter(VelFilt,ElabPar.NPVelFilt,DataOut{1, IT}.Data(:, ElabPar.ISign.XposEnc))/1000/DT; % gb: from micron to [mm/s]
        % DataOut{1, IT}.Data(:, ElabPar.ISign.XposEnc) = filtfilt(ones(ElabPar.NPSmoothFilt,1),ElabPar.NPSmoothFilt,XVel); % gb: additional low pass filter
        % DataOut{1, IT}.Data(:, ElabPar.ISign.OperVel) = filtfilt(ones(ElabPar.NPAccFilt,1),ElabPar.NPAccFilt,[0; diff(DataOut{1, IT}.Data(:,ElabPar.ISign.OperVel))]); % Operatin Wheel acceleration
        % % Current X1
        % ElabPar.ISign.X1IQ= 8;
        % DataOut{1, IT}.Data(:, ElabPar.ISign.X1IQ) = filtfilt(ones(ElabPar.NPSmoothFilt,1),ElabPar.NPSmoothFilt,DataOut{1, IT}.Data(:, ElabPar.ISign.X1IQ));
    end


end
end