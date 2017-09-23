function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

function retval = checkResult(y)
  if (y == 1)
    retval = 'k+';
  else
    retval = 'yo';
  endif
 endfunction
 
% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
for i = 1 : size(X,1)
plot(X(i,1), X(i,2), checkResult(y(i)), 'markersize', 10);
endfor



% =========================================================================



hold off;

end
