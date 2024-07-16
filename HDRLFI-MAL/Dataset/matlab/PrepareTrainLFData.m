clc
clear all;

%% path
folder = 'L:/Ghost_free_HDRLFI/Dataset/Final_DynamicSet_version/TrainSet/';
scene_inf = dir(folder);
scene_inf = scene_inf(3:end);
train_number = length(scene_inf);

%% parameters
angRes = 7;
patchsize = 96;
stride = 48;

save_path = ['../TrainingData_', num2str(angRes), 'x', num2str(angRes), '/'];
if ~exist(save_path,'dir')
    mkdir(save_path);
end

%% training dataset
idx = 0;
for iScene = 1:train_number
    in_expo1 = imread([folder,scene_inf(iScene).name,'/expo1.png']);
    in_expo2 = imread([folder,scene_inf(iScene).name,'/expo2.png']);
    in_expo3 = imread([folder,scene_inf(iScene).name,'/expo3.png']);
    
    [LFH, LFW, LFC] = size(in_expo2);
    in_expo1 = permute(reshape(in_expo1, [angRes,LFH/angRes,angRes,LFW/angRes,3]), [2,4,5,1,3]);   % [h,w,3,ah,aw]   
    in_expo2 = permute(reshape(in_expo2, [angRes,LFH/angRes,angRes,LFW/angRes,3]), [2,4,5,1,3]);   % [h,w,3,ah,aw]    
    in_expo3 = permute(reshape(in_expo3, [angRes,LFH/angRes,angRes,LFW/angRes,3]), [2,4,5,1,3]);   % [h,w,3,ah,aw]
    [H, W, ~, U, V] = size(in_expo2);

    in_hdri = hdrread([folder,scene_inf(iScene).name,'/HDRLFimg.hdr']);
    in_hdri = permute(reshape(in_hdri, [angRes,LFH/angRes,angRes,LFW/angRes,3]), [2,4,5,1,3]);     % [h,w,3,ah,aw]
    
    in_ev = single(load([folder,scene_inf(iScene).name,'/exposure.txt']));      % [3,1]
    
    for h = 1 : stride : H-patchsize+1
        for w = 1 : stride : W-patchsize+1      
            data1 = in_expo1(h:h+patchsize-1, w:w+patchsize-1, :, :, :);    % [ph,pw,3,ah,aw] 
            data2 = in_expo2(h:h+patchsize-1, w:w+patchsize-1, :, :, :);    % [ph,pw,3,ah,aw] 
            data3 = in_expo3(h:h+patchsize-1, w:w+patchsize-1, :, :, :);    % [ph,pw,3,ah,aw] 
            label = in_hdri(h:h+patchsize-1, w:w+patchsize-1, :, :, :);     % [ph,pw,3,ah,aw] 
            
            maxTH = 0.7;
            minTh = 0.3;
            buffer = im2single(squeeze(data2(:,:,:,1,1)));
            thresh = patchsize * patchsize;
            badInds = buffer > maxTH | buffer < minTh;
            inds = sum(sum(sum(badInds, 1), 2), 3) > thresh;
            
            if (inds) 
                % save data
                idx = idx + 1;
                SavePath_H5 = [save_path, num2str(idx,'%06d'),'.h5'];
                if exist(SavePath_H5,'file')
                    fprintf('Warning: replacing existing file %s \n',SavePath_H5);
                    delete(SavePath_H5);
                end    
                % overturn
                data1 = permute(data1,[3,2,1,5,4]);     % [ph,pw,3,ah,aw] --> [3,pw,ph,aw,ah]
                data2 = permute(data2,[3,2,1,5,4]);     % [ph,pw,3,ah,aw] --> [3,pw,ph,aw,ah]
                data3 = permute(data3,[3,2,1,5,4]);     % [ph,pw,3,ah,aw] --> [3,pw,ph,aw,ah]
                label = permute(label,[3,2,1,5,4]);     % [ph,pw,3,ah,aw] --> [3,pw,ph,aw,ah]
                ev_data = permute(in_ev, [2,1]);        % [3,1]--->[1,3]
                h5create(SavePath_H5, '/data1', size(data1), 'Datatype', 'uint16');
                h5write(SavePath_H5, '/data1', uint16(data1));
                h5create(SavePath_H5, '/data2', size(data2), 'Datatype', 'uint16');
                h5write(SavePath_H5, '/data2', uint16(data2));
                h5create(SavePath_H5, '/data3', size(data3), 'Datatype', 'uint16');
                h5write(SavePath_H5, '/data3', uint16(data3));         
                h5create(SavePath_H5, '/label', size(label), 'Datatype', 'single');
                h5write(SavePath_H5, '/label', single(label));   
                h5create(SavePath_H5, '/ev_data', size(ev_data), 'Datatype', 'single');
                h5write(SavePath_H5, '/ev_data', single(ev_data));      
            else
                continue;
            end
        end
    end
    fprintf('Processing on the Scene "%s"\n', num2str(iScene));  
end