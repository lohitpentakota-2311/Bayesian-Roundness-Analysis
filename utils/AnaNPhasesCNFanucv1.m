function [DataOut, DoET] = AnaNPhasesCNFanucv1(FileNameNC, ISign, MachProc, ElabPar, DoET,IT)

% Analizes signals measured by a FANUC Numerical Control during the execution of a typical Monzesi grinding cycle where
% material removal occurs in 4 phases (with different infeed velocities) + spark out.
% The following signals must be available: see the definition of ISign.* here below.
%
% INPUTS:
%       FileNameNC: full file name of the CSV data file created by FANUC Servoguide
%       ISign: indices of the signals acquired by the NC. It MUST contain:
%           .XposEnc: X axis position linear sensor
%           .XIQ: Iq current X axis
%           .ContrVel: control wheel velocity [motor rpm]
%           .ContrIQ: control wheel Q current [I peak]
%           .OperPow: operating wheel power (or, alerantively, .OperTorq for the % torque)
%           .OperVel: operating wheel speed
%       MachProc: structure with machine and process info
%           .ContrW.*: data on the Control wheel: .Vel_grind; .NC_Vel_grind
%           .OperW.*: data on the operating wheel:.OperW.TrRatio, .ContTorque;  Nm.NC_Vel_grind
%           .WP.*: data on the workpiece
%           .SparkoutTime = 0.2;
%           .OperW.EffIdle = 0.255; % operating wheel Effort [?] when rotating idle
%           .OperW.EffFluid = 0.3; % rough estimate of operating wheel effort [?] when getting near the workpiece (80-200um)
%           .OperW.MultCoeff = 4278; MachProc.OperW.AddCoeff = -148; % coeff to convert LMDAT signal into [W]
%           .Gamma = FixedT.Gamma_deg; % Blade angle [deg] (30: stable; 20 unstable)
%           .ContrW.TrRatio = 14.33;    % motor vel/ control wheel vel
%           .ContrW.DynEff = 0.7;   % dynamic efficiency of the Wormgear transmission (hypothesis)
%
%       ElabPar: structure with parameters for signal processing
%           .XDisplTol = 1;    % displacement tolerance to identify grinding stages in AnaNPhasesCNFanuc.m [micron]
%           .trigXVelPC: trigger threshold ofr XVel, percentange of Max Vel
%           .trigEff = 0.3; % threshold on the increase of the operating wheel effort signal
%           .trigEffPretrig = 0.2; % relative pretrigger internval on the operating wheel effort signal
%           .MaxRelErr: maximum relative error between planned and actual data from NC
%           .NPVelFilt: number of points for speed calculation filter (e.g. =25)
%           .NPSmoothFilt: number of points for generic signal smoothing, e.g. 100
%           .NPSmoothFilt2:  number of points for the generic signal smoothing (used on Operating wheel effort)
%           .FPlot: plot flag: if ==1 o 'yes', plots are made
%           .FPrint: if ==1 o 'yes', plots are printed to file
%       DoET: input & output table,  read& written to excel file
%       IT: test number in DoE

% OUPUTS:
%       DataOut: data read from the FANUC file
%       DoET: add elaborated data, containing:   (is EMPTYif file not found)
%           .NC_XvarNoload(IT): variance of X axis position in non-working zones [um]
%           .NC_XVelGrind(IT): mean X axis feed during grinding [mm/s]
%           .NC_ContrWVelGrind(IT): mean speed of control wheel during grinding [rpm]
%           .NC_WPVelGrind(IT): mean speed of workpiece during grinding [rps]
%           .NC_OperWVelGrind(IT): mean speed of operating wheel during grinding [rpm]
%           .NC_IndXretract(IT): index of the start of X axis retract movement []
%
% GB converts LMDAT into [W] and Nm
% 21/02/23; 01/03/23; LP 10/03/23; GB 14/03/23: ISparkOut; GB 20/03/23: OW power spectrum;
% 07/04/23 OW torque phase

[~,FileName, ~] = fileparts(FileNameNC); % separate file name parts
TitlePr = replace(FileName,'_',' ');        % string for titles

% loads NC signals:
% function [DataOut, PathName] = ReadFANUCcsv(FileNames,ElabPar);
[DataOut, ~] = ReadFANUCcsv([FileNameNC '.csv'],ElabPar); % posizione riga X in DECIMI di micron!!!
% % % Signal labels checking (26/04/2025)
expectedLabels = {
    ElabPar.Label.XposEnc1,
    ElabPar.Label.X1posEnc2,
    ElabPar.Label.ContrVel3,
    ElabPar.Label.ContrIQ4,
    ElabPar.Label.XIQ5,
    ElabPar.Label.OperTorq6,
    ElabPar.Label.OperVel7,
    ElabPar.Label.X1IQ8
    };

for i = 1:8
    if ~strcmp(DataOut.SignalLabels{i}, expectedLabels{i})
        error('Signal label %d does not match the expected value.', i);
    end
end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

if isempty(DataOut)
    DataOut = [];
else
    DataOut.Time = DataOut.Time/1000; % time in [s]
    DT = DataOut.Time(2); % [s]
    SamplFreq = 1/DT;
    NP = length(DataOut.Data(:,1));

    % nelle misure fatte sembra che la posizione riga X sia espressa in decimi di micron e non micron (dipende dai parametri inseriti in Servoguide)
    % non so sei sia così anche per la posizione X1. Per adesso correggo solo posiz X (e non X1, che non ha riga ottica)
    DataOut.Data(:,ISign.XposEnc) = DataOut.Data(:,ISign.XposEnc)/10;  % [um]   
    % Data processing

    % it uses a filter with a step profile to compute a smooth velocity signal from the X position signal
    % ElabPar.NPVelFilt : number of points for filtered vel calculation
    VelFilt = [ones(ElabPar.NPVelFilt,1); -ones(ElabPar.NPVelFilt,1)]/ElabPar.NPVelFilt;
    XVel = filter(VelFilt,ElabPar.NPVelFilt,DataOut.Data(:,ISign.XposEnc))/1000/DT; % from micron to [mm/s]
    XVelFilt = filtfilt(ones(ElabPar.NPSmoothFilt,1),ElabPar.NPSmoothFilt,XVel); % additional low pass filter
    OWAcc = filtfilt(ones(ElabPar.NPAccFilt,1),ElabPar.NPAccFilt,[0; diff(DataOut.Data(:,ElabPar.ISign.OperVel))]); % Operatin Wheel acceleration

    XVelThresh = ElabPar.trigXVelPC*max(abs(XVelFilt));
    IXbeginStop = find(abs(XVelFilt)>XVelThresh,1,'last')+floor(NP/20); % index of last section: X axis still, after grinding    % GB: (abs(
    % in the standard grinding cycle there is a rapid advance (speed<0), then the low speed motion, splitted in N phases,
    % then a rapid retraction (speed>0):
    if isnan(IXbeginStop)||IXbeginStop>NP
        IXbeginStop=NP-50; warning([TitlePr ' final Xstopped interval not found']);
    end

    % centerless grinding kinematics:
%     AngBetaS = asin(DoET.hw_mm(IT)/(MachProc.OperW.Diam+MachProc.WP.Diam)); % angle horiz/center OperWheel-center WP
%     AngBetaC = asin(DoET.hw_mm(IT)/(MachProc.ContrW.Diam+MachProc.WP.Diam)); % angle horiz/center ContrWheel-center WP
%     AngAlpha = pi/2-MachProc.Gamma/180*pi-AngBetaS;
%     AngBeta = AngBetaS+AngBetaC; % Sum internal angles
%     K1 = sin(AngBeta)/sin(AngAlpha+AngBeta); K2 = sin(AngAlpha)/sin(AngAlpha+AngBeta); % coefficeint for the regenrative equation

    % manages measurements with Operating wheel power or torque:  NB: only one of them must be present in ISign!!
    if isfield(ISign,'OperPow')
        OperWheelWatt = MachProc.OperW.MultCoeff*DataOut.Data(:,ISign.OperPow)+MachProc.OperW.AddCoeff; % converted in W and into Nm
        OWtorque = OperWheelWatt./(DataOut.Data(:,ISign.OperVel)*2*pi/60);  % torque [Nm]
    elseif isfield(ISign,'OperTorq')
        OWtorque = DataOut.Data(:,ISign.OperTorq)/100*MachProc.OperW.ContTorque*MachProc.OperW.TrRatio; % torque [Nm] at the operating wheel
    else
        ISign
        error('Operating wheel effort signal missing!')
    end
    % estimates the torque at the operating wheel from the motor torque, removing inertia effects: (it seems it's not very important)
    OWTorqueComp = OWtorque-MachProc.OperW.InertComp*OWAcc*MachProc.OperW.TrRatio;
    OWTorqueCompFilt = filtfilt(ones(ElabPar.NPSmoothFilt2,1),ElabPar.NPSmoothFilt2,OWTorqueComp);

    % filters the operating effort to attenuate the XX oscillations:
    OWtorqueFilt = filtfilt(ones(ElabPar.NPSmoothFilt2,1),ElabPar.NPSmoothFilt2,OWtorque);

    % identifies the grinding interval by the theoretical X axis displacement (not including the initial air cutting):
    NPhases = DoET.Number_of_stages(IT);
    PhasesFeedDispl = zeros(NPhases,1); PhasesMeanCWIQ = zeros(NPhases,1); 
    PhasesStartIndex =  zeros(NPhases+1,1); PhasesEndIndex =  zeros(NPhases+1,1); PhasesMeanOWtorque = zeros(NPhases,1);
    for IS = 1:NPhases
        PhasesFeedDispl(IS) = DoET.(sprintf('s%uFeed_micron',IS))(IT);
    end
    PhasesFinalX = min(DataOut.Data(:,ISign.XposEnc))+ flip(cumsum([0; flip(PhasesFeedDispl)])); % final X axis position of each phase, including the initial air cutting

    % analizes the Air Cutting phase (where it's supposed that no actual grinding occurs):
%     ICWGrindSpeedBeforeGrind = round(IStartCWGrindSpeed+0.2*(IstartGrinding-IStartCWGrindSpeed)):round(IstartGrinding-0.2*(IstartGrinding-IStartCWGrindSpeed)); % data index interval at grinding speed, before actual grinding
%     % force estimation from the control wheel current:
%     DoET.NC_ContrWIQnoGrind(IT) = round(mean(DataOut.Data(ICWGrindSpeedBeforeGrind,ISign.ContrIQ)),4); % mean control wheel current [% nominal, peak] at grinding speed, before actual cutting

    for IS = 0:NPhases
        if IS ==0 % air cutting
            % saves the corresponding data after the last stage:
            PhasesEndIndex(NPhases+1) = find(DataOut.Data(:,ISign.XposEnc)<(PhasesFinalX(1)+ElabPar.XDisplTol),1);
            PhasesStartIndex(NPhases+1) = find(DataOut.Data(:,ISign.XposEnc)<(PhasesFinalX(1)+DoET.AirCutting_Feed_micron(IT)-ElabPar.XDisplTol),1); % it's computed backward from the final value and the phase displacement, taking into account the tolerance
            IPhase = PhasesStartIndex(NPhases+1):PhasesEndIndex(NPhases+1);
            DoET.(sprintf('NC_AirCuttFeedVel_micron_s'))(IT) = abs(mean(XVelFilt(IPhase))*1000);
            CheckErr(TitlePr,ElabPar,'Air Cutting. Mean feed speed. Theo: %4.1f, NC: %4.1f [um/s]', DoET.(sprintf('AirCutting_FeedVel_micron_s'))(IT), DoET.(sprintf('NC_AirCuttFeedVel_micron_s'))(IT));
            DoET.(sprintf('NC_AirCuttingCW_rpm'))(IT) = abs(mean(DataOut.Data(IPhase,ISign.ContrVel)))/MachProc.ContrW.TrRatio;
            CheckErr(TitlePr,ElabPar,'AirCutting. Mean CW speed. Theo: %4.1f, NC: %4.1f [rpm]', DoET.(sprintf('AirCutting_CW_rpm'))(IT), DoET.(sprintf('NC_AirCuttingCW_rpm'))(IT));
            NC_ContrWIQnoGrind =  mean(DataOut.Data(IPhase,ISign.ContrIQ));
            NC_OperWtorqnoGrind = mean(OWtorque(IPhase)); DataOut.InoGrind = IPhase;
            DoET.(sprintf('NC_AirCuttTorqueOW_Nm'))(IT) = NC_OperWtorqnoGrind;
            NC_XIQnoGrind = mean(DataOut.Data(IPhase,ISign.XIQ));
        else
            PhasesEndIndex(IS) = find(DataOut.Data(:,ISign.XposEnc)<PhasesFinalX(IS+1)+ElabPar.XDisplTol,1);
            PhasesStartIndex(IS) = find(DataOut.Data(:,ISign.XposEnc)<PhasesFinalX(IS+1)+PhasesFeedDispl(IS)-ElabPar.XDisplTol,1); % it's computed backward from the final value and the phase displacement, taking into account the tolerance
            IPhase = PhasesStartIndex(IS):PhasesEndIndex(IS);
            DoET.(sprintf('NC_s%uFeedVel_micron_s',IS))(IT) = abs(mean(XVelFilt(IPhase)))*1000; % [um/s]
            CheckErr(TitlePr,ElabPar,[sprintf('stage n%u.',IS) ' Mean feed speed. Theo: %4.1f, NC: %4.1f [um/s]'], DoET.(sprintf('s%uFeedVel_micron_s',IS))(IT), DoET.(sprintf('NC_s%uFeedVel_micron_s',IS))(IT));
            DoET.(sprintf('NC_s%uCW_rpm',IS))(IT) = abs(mean(DataOut.Data(IPhase,ISign.ContrVel)))/MachProc.ContrW.TrRatio;
            CheckErr(TitlePr,ElabPar, [sprintf('stage n%u.',IS),' Mean CW speed. Theo: %4.1f, NC: %4.1f [rpm]'], DoET.(sprintf('s%uCW_rpm',IS))(IT), DoET.(sprintf('NC_s%uCW_rpm',IS))(IT));
            PhasesMeanCWIQ(IS) = mean(DataOut.Data(IPhase,ISign.ContrIQ)); % mean control wheel current [% nominal,peak]  in the phase
            PhasesMeanOWtorque(IS) = mean(OWtorque(IPhase));
            DoET.(sprintf('NC_s%uMeanXIQ_A',IS))(IT) =  mean(DataOut.Data(IPhase,ISign.XIQ));
            DoET.(sprintf('NC_s%uStdXIQ_A',IS))(IT) =  std(DataOut.Data(IPhase,ISign.XIQ));

            % the tangential force on the control wheel is estimated by the current increase during grinding, to remove the
            % frictional effects (and neglecting more complex effects), taking into account of the transmisison ratio and the wheel
            % radius [mm]. The torque constant refers Nm to the RMS current, while the measured current is the peak value: to be divided by sqrt(2):
            DoET.(sprintf('NC_s%uCWTanForce',IS))(IT) = round(abs(PhasesMeanCWIQ(IS)-NC_ContrWIQnoGrind)*MachProc.ContrW.INom/100/sqrt(2)*...
                MachProc.ContrW.Kt*MachProc.ContrW.TrRatio/MachProc.ContrW.DynEff*2000/MachProc.ContrW.Diam,4);

            % the tangential force on the operating wheel is estimated by the torque increase during grinding, to remove the
            % frictional effects (neglecting more complex effects):
            DoET.(sprintf('NC_s%uOWTanForceGrind',IS))(IT) =  round((PhasesMeanOWtorque(IS)-NC_OperWtorqnoGrind)*2000/MachProc.OperW.Diam,4);

            % estimation of the force along X:
            DoET.(sprintf('NC_s%uXForceGrind',IS))(IT) = round(abs(DoET.(sprintf('NC_s%uMeanXIQ_A',IS))(IT)-NC_XIQnoGrind)*MachProc.X.INom/100/sqrt(2)*MachProc.X.Kt/MachProc.X.DynEff*2*pi*1000/MachProc.X.Lead,4);
            DoET.(sprintf('NC_s%uFtan_FX_ratio',IS))(IT) = round(DoET.(sprintf('NC_s%uOWTanForceGrind',IS))(IT)/DoET.(sprintf('NC_s%uXForceGrind',IS))(IT),4);
        end
    end
end

Igrinding = PhasesStartIndex(1):PhasesEndIndex(end-1);    % total grinding interval (excluding air cutting)
DataOut.IGrinding = Igrinding;
DoET.NC_OperWVelGrind(IT) = round(mean(abs(DataOut.Data(Igrinding,ISign.OperVel)))/MachProc.OperW.TrRatio,4);
CheckErr(TitlePr,ElabPar,'mean grinding wheel speed during grinding. Theo: %4.1f, NC: %4.1f [rpm]',MachProc.OperW.SpeedRPM, DoET.NC_OperWVelGrind(IT));

ISparkoutEnd = find(XVelFilt(Igrinding(end):end)>0,1);  % identifies the instant when the X axis starts the rapid opening motion
DataOut.ISparkOut = Igrinding(end)+(1:ISparkoutEnd);

% [Freq, STF] = fftg(Time, Sign, Zer)
[FreqXvelGrind, FFTXvelGrind] = fftg(DataOut.Time(Igrinding), XVel(Igrinding), 2*length(Igrinding)); % zero padding to increase frequency resolution
[FreqXiqGrind, FFTXiqGrind] = fftg(DataOut.Time(Igrinding), detrend(DataOut.Data(Igrinding,ISign.XIQ),1), 2*length(Igrinding));

% zero padding to increase frequency resolution:
[FreqOWtorGrind, FFTOWtorGrind] = fftg(DataOut.Time(Igrinding(1):(Igrinding(end)+ISparkoutEnd)), detrend(OWtorque(Igrinding(1):(Igrinding(end)+ISparkoutEnd)),1) ...
    , 2*(ISparkoutEnd+Igrinding(end)-Igrinding(1)+1));
[MaxFFTOWtorGrindSO,IndMaxFFTOWtorGrindSO] = max(abs(FFTOWtorGrind(2:end))); % the 0Hz is skipped
RMSFFTOWtorGrindSO = rms(FFTOWtorGrind(2:min(length(FFTOWtorGrind),10*IndMaxFFTOWtorGrindSO))); % it evaluates the "average amplitude" of the spectrum

DoET.NC_OperWTorMaxFFTHz(IT) = round(FreqOWtorGrind(1+IndMaxFFTOWtorGrindSO),4);
DoET.NC_OperWTorMax_rmsFFT(IT) = round(MaxFFTOWtorGrindSO/RMSFFTOWtorGrindSO,4);
% it refines the peak location by second order interpolation:
% function [YPeak, XPeak] = InterpPeak(XVect,YVect,IndexPeak,MaxMin,LatNPointsInterp)
LatNPointsInterp = 2; % points around the max used for quadratic fitting
[~, FrPeak] = InterpPeak(FreqOWtorGrind,abs(FFTOWtorGrind),IndMaxFFTOWtorGrindSO,1,LatNPointsInterp);
DoET.NC_OperWTorMaxFFTHzInterp(IT) = round(FrPeak,4);
IndFitting = IndMaxFFTOWtorGrindSO+(-LatNPointsInterp:LatNPointsInterp);
DoET.NC_OperWTorMaxFFTPhaseDegInterp(IT) = round(interp1(FreqOWtorGrind(IndFitting),180/pi*angle(FFTOWtorGrind(IndFitting)),FrPeak),4);

% Plots:
if strcmpi(ElabPar.FPlot,'YES')||(isnumeric(ElabPar.FPlot)&&ElabPar.FPlot==1)||strcmpi(ElabPar.FPrint,'YES')||(isnumeric(ElabPar.FPrint)&&ElabPar.FPrint==1)
    if strcmpi(ElabPar.FPlot,'YES')||(isnumeric(ElabPar.FPlot)&&ElabPar.FPlot==1) % control plot visibility
        VisiFlag = 'on';
    else
        VisiFlag = 'off';
    end
    if 0
    Descr = [TitlePr ' effort histogram'];
    f= figure('Name',Descr,'Visible',VisiFlag); histogram(OWtorqueFilt,NBins); % oper wheel effort histogram
    title(Descr);
    saveFigF(f, ElabPar, Descr)

    Descr = [TitlePr ' X axis'];
    f= figure('Name',Descr,'Visible',VisiFlag);
    ax1 = subplot(311); plot(DataOut.Time,DataOut.Data(:,ISign.XposEnc)); grid on; title([TitlePr ': X axis signals']);...
        ylabel('X axis pos [um]');
    ax2 = subplot(312); plot(DataOut.Time,[XVel XVelFilt]); grid on; ylabel('X axis vel [mm/s]');
    ax3 = subplot(313); yyaxis left; plot(DataOut.Time,DataOut.Data(:,ISign.XIQ)); grid on; xlabel('time [s]'); ylabel('X IQ');...
        yyaxis right; plot(DataOut.Time,OWtorque); grid on; ylabel('OperEffort [?]'); hold on; ...
        plot(DataOut.Time(Igrinding),OWtorque(Igrinding),'-','LineWidth',2); hold off;
    linkaxes([ax1 ax2 ax3],'x');
    % function saveFigF(FigHead, ElabPar, FName)
    saveFigF(f, ElabPar, Descr)
    end

    Descr = [TitlePr ' stage definition'];
    f= figure('Name',Descr,'Visible',VisiFlag);
    bx1 = subplot(311); plot(DataOut.Time,OWtorque(:)); grid on; ylabel('OperEffort [?]'); title(Descr,'Interpreter','none');
    bx2 = subplot(312); plot(DataOut.Time,DataOut.Data(:,ISign.XposEnc)); grid on; ylabel('X pos [um]');
    bx3 = subplot(313); plot(DataOut.Time,XVelFilt); grid on; xlabel('time [s]'); ylabel('X vel []');
    hold on;
    Leg = {'vel X'};
    for IS = 1: (NPhases+1)
        IPhase = PhasesStartIndex(IS):PhasesEndIndex(IS);
        plot(DataOut.Time(IPhase),XVelFilt(IPhase),'LineWidth',2);
        if IS>NPhases
            Leg{IS+1} = 'air cutting'; % last data refer to air cutting
        else
            Leg{IS+1} = sprintf('stage: %u',IS); %
        end
    end
    plot(DataOut.Time(DataOut.ISparkOut),XVelFilt( DataOut.ISparkOut),'LineWidth',2);
    Leg{IS+2} ='sparkout';
    hold off; legend(Leg,'location','east');
    linkaxes([bx1 bx2 bx3],'x');
    saveFigF(f, ElabPar, Descr)

    Descr = [TitlePr ' effort signals'];
    f= figure('Name',Descr,'Visible',VisiFlag);
    bx1 = subplot(311); plot(DataOut.Time,OWtorque(:)); grid on; ylabel('OWtorque [Nm]'); title([TitlePr ': effort signals']);...
        bx2 = subplot(312); plot(DataOut.Time,DataOut.Data(:,ISign.ContrIQ)); grid on; ylabel('IQ control W [A]');
    bx3 = subplot(313); plot(DataOut.Time,DataOut.Data(:,ISign.XIQ)); grid on; xlabel('time [s]'); ylabel('X IQ');
    hold on;
    Leg = {'IQX'};
    for IS = 1: (NPhases+1)
        IPhase = PhasesStartIndex(IS):PhasesEndIndex(IS);
        subplot(311); hold on;  plot(DataOut.Time(IPhase),OWtorque((IPhase)),'LineWidth',2); hold off;
        subplot(312); hold on;  plot(DataOut.Time(IPhase),DataOut.Data(IPhase,ISign.ContrIQ),'LineWidth',2); hold off;
        subplot(313); hold on;   plot(DataOut.Time(IPhase),DataOut.Data(IPhase,ISign.XIQ),'LineWidth',2); hold off;
        if IS>NPhases
            Leg{IS+1} = 'air cutting'; % last data refer to air cutting
        else
            Leg{IS+1} = sprintf('stage: %u',IS); %
        end
    end
    subplot(311); hold on;  plot(DataOut.Time(DataOut.ISparkOut),OWtorque(DataOut.ISparkOut),'LineWidth',2); hold off;
    subplot(312); hold on;  plot(DataOut.Time(DataOut.ISparkOut),DataOut.Data(DataOut.ISparkOut,ISign.ContrIQ),'LineWidth',2); hold off;
    subplot(313); hold on;   plot(DataOut.Time(DataOut.ISparkOut),DataOut.Data(DataOut.ISparkOut,ISign.XIQ),'LineWidth',2); hold off;
    Leg{IS+2} ='sparkout';

    legend(Leg,'location','east');
    linkaxes([bx1 bx2 bx3],'x');
    saveFigF(f, ElabPar, Descr)

    % X spectra during grinding:
    Descr = [TitlePr ' CN FFT'];
    f= figure('Name',Descr,'Visible',VisiFlag);
    yyaxis left; plot(FreqXvelGrind,abs(FFTXvelGrind)); grid on; xlabel('freq [Hz]'); ylabel('X vel spectr');...
        yyaxis right; plot(FreqXiqGrind,abs(FFTXiqGrind)); grid on; ylabel('X torque [Nm]');  title([TitlePr ': FFT']);...
        saveFigF(f, ElabPar, Descr)
    Descr = [TitlePr ' CN OWtorFFT'];
        f= figure('Name',Descr,'Visible',VisiFlag);
    plot(FreqOWtorGrind, abs(FFTOWtorGrind)); grid on; xlabel('freq [Hz]'); ylabel('OW torque spectr'); title([TitlePr ': CN OWtorFFT']);...
        saveFigF(f, ElabPar, Descr)
end
end

function CheckErr(TitleString,ElabPar,FormString,RefVal, ActVal)
% checks the relative error between two values

if abs((ActVal-RefVal)/RefVal)>ElabPar.MaxRelErr
    warning('off','backtrace');
    warning(['Test %s, ' FormString '\n'],TitleString,RefVal, ActVal)
    warning('on','backtrace')
else
    fprintf(['Test %s, ' FormString '\n'],TitleString,RefVal, ActVal)
end
end