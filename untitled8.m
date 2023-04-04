% Detect feature points in the left image
imagePoints1 = detectMinEigenFeatures(rgb2gray(Limg), 'MinQuality', 0.1);

% Create the point tracker
tracker = vision.PointTracker('MaxBidirectionalError', 1, 'NumPyramidLevels', 5);

% Initialize the point tracker
imagePoints1 = imagePoints1.Location;
initialize(tracker, imagePoints1, Limg);

% Track the points in the right image
[imagePoints2, validIdx] = step(tracker, Rimg);
matchedPoints1 = imagePoints1(validIdx, :);
matchedPoints2 = imagePoints2(validIdx, :);

% Estimate the fundamental matrix
intrinsics = stereoParams3.CameraParameters1;
[F, epipolarInliers] = estimateFundamentalMatrix(matchedPoints1, matchedPoints2, 'Method', 'Norm8Point', 'NumTrials', 4000, 'DistanceThreshold', 0.1, 'Confidence', 99.99);

% Find epipolar inliers
inlierPoints1 = matchedPoints1(epipolarInliers, :);
inlierPoints2 = matchedPoints2(epipolarInliers, :);

% Display inlier matches
figure
showMatchedFeatures(Limg, Rimg, inlierPoints1, inlierPoints2);
title('Epipolar Inliers');

% Estimate relative pose
[E,~] = cameraPose(F, stereoParams3.CameraParameters1, inlierPoints1, inlierPoints2);

% Release the tracker object
release(tracker);

% Detect dense feature points in the left image
border = 30;
roi = [border, border, size(Limg, 2)- 2*border, size(Limg, 1)- 2*border];
imagePoints1 = detectMinEigenFeatures(rgb2gray(Limg), 'ROI', roi, 'MinQuality', 0.001);

% Re-create and initialize the point tracker
tracker = vision.PointTracker('MaxBidirectionalError', 1, 'NumPyramidLevels', 5);
imagePoints1 = imagePoints1.Location;
initialize(tracker, imagePoints1, Limg);

% Track the points in the right image
[imagePoints2, validIdx] = step(tracker, Rimg);
matchedPoints1 = imagePoints1(validIdx, :);
matchedPoints2 = imagePoints2(validIdx, :);

% Compute the camera matrices for each position of the camera
camMatrix1 = cameraMatrix(intrinsics, eye(3), [0 0 0]);
camMatrix2 = cameraMatrix(intrinsics, relPose.RotationMatrix, relPose.Translation.');


% Compute the 3-D points
points3D = triangulate(matchedPoints1, matchedPoints2, cameraMatrix1, cameraMatrix2Array);

% Get the color of each reconstructed point
numPixels = size(Limg, 1) * size(Limg, 2);
allColors = reshape(Limg, [numPixels, 3]);
colorIdx = sub2ind([size(Limg, 1), size(Limg, 2)], round(matchedPoints1(:,2)), round(matchedPoints1(:,1)));
color = allColors(colorIdx, :);


% Visualize the reconstructed 3-D points
figure
pcshow(points3D, color, 'VerticalAxis', 'Y', 'VerticalAxisDir', 'down', 'MarkerSize', 45);
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Reconstructed 3-D Points');

% Save the point cloud to a PLY file
pcwrite(pointCloud(points3D, 'Color', color), 'pointcloud.ply');

