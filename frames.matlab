vidobj=VideoReader('timelapsen.avi');
frames=vidobj.Numberofframes;
for f=1:frames
  thisframe=read(vidobj,f);
  figure(1);imagesc(thisframe);
  thisfile=sprintf('/Users/[redacted]/Desktop/img_data/raw/2_frame_%03d.jpg',f);
  imwrite(thisframe,thisfile);
end