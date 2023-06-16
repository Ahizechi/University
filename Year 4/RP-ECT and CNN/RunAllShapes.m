% RunAllShapes.m

close all; clear; clc;

% Start the timer for total runtime
tic;

% Number of scripts to run
num_scripts = 13;

scripts = {'SpeedArrow.mlx', 'SpeedCross.mlx', 'SpeedDiamond.mlx', 'SpeedEllipsoid.mlx', ...
    'SpeedHeart.mlx', 'SpeedHexagon.mlx', 'SpeedPentagon.mlx', 'SpeedRectangle.mlx', ...
    'SpeedSquare.mlx', 'SpeedTri.mlx', 'AddNoise.mlx', 'SpeedEnhance.mlx', ...
    'SaveNoiseFiltered.mlx'};

% Create a parallel pool with 5 workers if it does not exist
if isempty(gcp('nocreate'))
    parpool(5);
end

% Initialize a variable to store the elapsed times
elapsed_times = zeros(num_scripts, 1);

% Run the first 10 scripts in parallel
parfor idx = 1:10
    % Start the timer for the current script
    script_tic = tic;

    % Run the script
    script_name = scripts{idx};
    run_script(script_name); % Call the run_script function here

    % Store the elapsed time
    elapsed_times(idx) = toc(script_tic);
end

% Run the 11th script by itself
script_tic = tic;
script_name = scripts{11};
run_script(script_name);
elapsed_times(11) = toc(script_tic);

% Run the final 2 scripts in parallel
parfor idx = 12:13
    % Start the timer for the current script
    script_tic = tic;

    % Run the script
    script_name = scripts{idx};
    run_script(script_name); % Call the run_script function here

    % Store the elapsed time
    elapsed_times(idx) = toc(script_tic);
end

% Calculate and display total runtime
total_runtime = toc;
fprintf('Total runtime: %.2f seconds\n', total_runtime);

% Save elapsed times to a .txt file
output_file = 'D:\University\Year 4\Project\Simulation\times.txt';
fileID = fopen(output_file, 'w');
for idx = 1:num_scripts
    fprintf(fileID, 'Script %d: %.2f seconds\n', idx, elapsed_times(idx));
end
fclose(fileID);
