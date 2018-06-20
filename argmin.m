function  smin=argmin(a)
              [m i]=min(a(:));
              smin=ind2sub(size(a),i);
end