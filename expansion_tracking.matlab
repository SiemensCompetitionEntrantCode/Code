% Blastocoel cavity expansion CSA tracking algorithm
% using Sobel operators and image segmentation techniques.
% Expansion slope was calculated from the regression line.
% A plot of CSA against time can also be called.

% Instantiate a video reader object to reduce the
% processing capacity to analyze the time-lapse images.
videoFReader = vision.VideoFileReader('aneuploid1_cut.mov');
depVideoPlayer = vision.DeployableVideoPlayer;

% Create structuring elements to assist the pixel dilation
% and image eroding functions. These values for the elements
% were determined empirically. 
se90 = strel('line', 33, 90);
se0 = strel('line', 33, 0);
seD = strel('diamond',1);

% Initiate the numerical or vectoral objects that will 
% yield numerical values for later operations.
index = 0
indicies = []
csas = []
ii = 1

% Loop over each frame of the video reader object
% and perform a variety of image segmentation
% methods on each of the frames. 
cont = ~isDone(videoFReader);
  while cont
    frame = step(videoFReader);
    frame2 = rgb2gray(frame);
    frame2 = edge(frame2, 'Sobel');
    frame2 = imdilate(frame2, [se0 se90]);
    % Remove all objects smaller than
    % 40000 pixels from imaage.
    frame2 = bwareaopen(frame2, 40000);
    % Fill the gaps in the segmented
    % image for a full segmentation.
    frame2 = imfill(frame2, 'holes');
    % Perform multiple erosions on 
    % the segmented image to remove
    % small irregularities.
    frame2 = imerode(frame2,se0);
    frame2 = imerode(frame2,se90);
    frame2 = imerode(frame2,seD);
    frame2 = imerode(frame2,seD);
    frame2 = imerode(frame2,seD);
    frame2 = imerode(frame2,seD);
    % Superimpose the segmented image over the
    % original image using an outline function
    BWoutline = bwperim(frame2, 26);
    frame(BWoutline) = 255; 
    % Update the indicies to keep track of
    % the number of frams being analyzed.
    index = index+1
    indicies(end+1) = index
    % Calculate the pixel area to determine the
    % CSA of the expanding blastocyst
    pixarea = bwarea(frame2)
    % Convert to micrometers
    micrometerarea = pixarea*((244^2)/(820^2))
    % Append current area in micrometers to a 
    % vector of all CSAs.
    csas(end+1) = micrometerarea
    % Optional: write frames to separate
    % directory for further analysis.
    imgname = sprintf('frame_%03d.png',ii)
    % Only call this line below if you want to write
    % all files to the working directory
    % imwrite(frame, imgname)
    ii = ii+1
    step(depVideoPlayer, frame);
    cont = ~isDone(videoFReader) && isOpen(depVideoPlayer);
  end

% Calculations for expansion slope using the time of
% expansion as the x-axis values instead of frame number.
timeofexp = 10   % the hours of expansion
incr = timeofexp/(indicies(end)-1)
xaxis = 0:incr:timeofexp
scatter(xaxis, csas)
coefs = polyfit(xaxis, csas, 1)
exp_slope = coefs(1)