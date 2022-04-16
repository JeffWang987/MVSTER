function None = BaseEvalMain_func(plyPath)

% clear all
% close all
format compact

% script to calculate distances have been measured for all included scans (UsedSets)

dataPath='/mnt/cfs/algorithm/public_data/mvs/dtu_evalset/SampleSet/MVS Data';
% pred_results='cascade_hr/48-32-8_4-2-1_dlossw-0.5-1.0-2.0_chs888/gipuma_4_0.9/';
% plyPath=['../../outputs/1101/dtu/' pred_results];

% plyPath = '/mnt/cfs/algorithm/xiaofeng.wang/jeff/code/MVS/cascade-stereo/CasMVSNet/outputs/repo_model_aligncorners_ITGT'
resultsPath=[plyPath '/eval_out/'];
disp(resultsPath);
mkdir(resultsPath);

method_string='mvsnet';
light_string='l3'; % l3 is the setting with all lights on, l7 is randomly sampled between the 7 settings (index 0-6)
representation_string='Points'; %mvs representation 'Points' or 'Surfaces'

switch representation_string
    case 'Points'
        eval_string='_Eval_'; %results naming
        settings_string='';
end

% get sets used in evaluation
UsedSets=[1 4 9 10 11 12 13 15 23 24 29 32 33 34 48 49 62 75 77 110 114 118];
% UsedSets=[15];

dst=0.2;    %Min dist between points when reducing

parfor cIdx=1:length(UsedSets)
    %Data set number
    cSet = UsedSets(cIdx)
    %input data name
    DataInName=[plyPath sprintf('/%s%03d_%s%s.ply',lower(method_string),cSet,light_string,settings_string)]

    %results name
    EvalName=[resultsPath method_string eval_string num2str(cSet) '.mat']

    %check if file is already computed
    if(~exist(EvalName,'file'))
        disp(DataInName);

        time=clock;time(4:5), drawnow

        tic
        Mesh = plyread(DataInName);
        Qdata=[Mesh.vertex.x Mesh.vertex.y Mesh.vertex.z]';
        toc

        BaseEval=PointCompareMain(cSet,Qdata,dst,dataPath);

        disp('Saving results'), drawnow
        toc
        mySave(EvalName, BaseEval);
        toc

        % write obj-file of evaluation
        % BaseEval2Obj_web(BaseEval,method_string, resultsPath)
        % toc
        time=clock;time(4:5), drawnow

        BaseEval.MaxDist=20; %outlier threshold of 20 mm

        BaseEval.FilteredDstl=BaseEval.Dstl(BaseEval.StlAbovePlane); %use only points that are above the plane
        BaseEval.FilteredDstl=BaseEval.FilteredDstl(BaseEval.FilteredDstl<BaseEval.MaxDist); % discard outliers

        BaseEval.FilteredDdata=BaseEval.Ddata(BaseEval.DataInMask); %use only points that within mask
        BaseEval.FilteredDdata=BaseEval.FilteredDdata(BaseEval.FilteredDdata<BaseEval.MaxDist); % discard outliers

        fprintf("mean/median Data (acc.) %f/%f\n", mean(BaseEval.FilteredDdata), median(BaseEval.FilteredDdata));
        fprintf("mean/median Stl (comp.) %f/%f\n", mean(BaseEval.FilteredDstl), median(BaseEval.FilteredDstl));
    end
end

end

function mySave(filenm, data)
    save(filenm, 'data');
end