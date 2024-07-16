clc
clear all;

%% path
folder = 'L:/Ghost_free_HDRLFI/Supplementary_experiments/Scenes/';
scene_inf = dir(folder);
scene_inf = scene_inf(3:end);
test_number = length(scene_inf);

%% parameters
angRes = 7;

save_path = ['../Other_TestData_', num2str(angRes), 'x', num2str(angRes), '/'];
if ~exist(save_path,'dir')
    mkdir(save_path);
end

%% test dataset
idx = 0;
for iScene = 1:test_number
    in_expo1 = uint8([]);  in_expo2 = uint8([]);  in_expo3 = uint8([]);
     % [h,w,3,ah,aw]   
    for p = 1:angRes
        for q = 1:angRes
            in_expo1(:,:,:,p,q) = imread([folder,scene_inf(iScene).name,'/expo1/',sprintf('%02d',p),'_',sprintf('%02d',q),'.png']);
            in_expo2(:,:,:,p,q) = imread([folder,scene_inf(iScene).name,'/expo2/',sprintf('%02d',p),'_',sprintf('%02d',q),'.png']);
            in_expo3(:,:,:,p,q) = imread([folder,scene_inf(iScene).name,'/expo3/',sprintf('%02d',p),'_',sprintf('%02d',q),'.png']);
        end
    end
    
    in_ev = [0;2;4];     % [3,1]
    
     % save data
    idx = idx + 1;
    SavePath_H5 = [save_path, num2str(idx,'%03d'),'.h5'];
    if exist(SavePath_H5,'file')
        fprintf('Warning: replacing existing file %s \n',SavePath_H5);
        delete(SavePath_H5);
    end  
    
    % overturn
    data1 = permute(in_expo1,[3,2,1,5,4]);     % [h,w,3,ah,aw] --> [3,w,h,aw,ah]
    data2 = permute(in_expo2,[3,2,1,5,4]);     % [h,w,3,ah,aw] --> [3,w,h,aw,ah]
    data3 = permute(in_expo3,[3,2,1,5,4]);     % [h,w,3,ah,aw] --> [3,w,h,aw,ah]
    ev_data = permute(in_ev, [2,1]);           % [3,1]--->[1,3]
    
    h5create(SavePath_H5, '/data1', size(data1), 'Datatype', 'uint8');
    h5write(SavePath_H5, '/data1', uint8(data1));
    h5create(SavePath_H5, '/data2', size(data2), 'Datatype', 'uint8');
    h5write(SavePath_H5, '/data2', uint8(data2));
    h5create(SavePath_H5, '/data3', size(data3), 'Datatype', 'uint8');
    h5write(SavePath_H5, '/data3', uint8(data3));         
    h5create(SavePath_H5, '/ev_data', size(ev_data), 'Datatype', 'single');
    h5write(SavePath_H5, '/ev_data', single(ev_data));      

    fprintf('Processing on the Scene "%s"\n', num2str(iScene));  
end