clc;
close all; 
clear;
workspace;
format long g;
format compact;
fontSize = 20;

folder = uigetdir(pwd, 'Select folder');
if folder == 0
	return;
end

imageFiles = [dir(fullfile(folder,'*.png'))];

numberOfImages = length(imageFiles);
theyreColorImages = false;
for k = 1 : numberOfImages
	fullFileName = fullfile(folder, imageFiles(k).name);
	fprintf('About to read %s\n', fullFileName);
	thisImage=imread(fullFileName);
	[thisRows, thisColumns, thisNumberOfColorChannels] = size(thisImage);
	if k == 1
		% Save the first image.
		sumImage = double(thisImage);
		% Save its dimensions so we can match later images' sizes to this first one.
		rows1 = thisRows;
		columns1 = thisColumns;
		numberOfColorChannels1 = thisNumberOfColorChannels;
		theyreColorImages = numberOfColorChannels1 >= 3;

    else
        if rows1 ~= thisRows || columns1 ~= thisColumns
			% It's not the same size, so resize it to the size of the first image.
			thisImage = imresize(thisImage, [rows1, columns1]);
		end
		sumImage = sumImage + double(thisImage); 
		
		if theyreColorImages
			displayedImage = uint8(sumImage / k);
		else
			displayedImage = sumImage;
		end
		imshow(displayedImage, []);
		drawnow;
	end
end

%--------------------------------------------------------------------------------
sumImage = uint8(sumImage / numberOfImages);
cla;
imshow(sumImage, []);
caption = sprintf('Average of %d Images', numberOfImages);
title(caption, 'FontSize', fontSize);

%--------------------------------------------------------------------------------
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
set(gcf, 'Toolbar', 'none', 'Menu', 'none');
set(gcf, 'Name', 'Demo by ImageAnalyst', 'NumberTitle', 'Off') 