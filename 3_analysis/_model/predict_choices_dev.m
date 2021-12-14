function [ dev ] = predict_choices_dev( p, y_emp, x)
%DIRPRIORDEV Summary of this function goes here
%   Detailed explanation goes here

y_pred = predict_choices(p,x);
dev = nansum((y_emp(:)-y_pred(:)).^2);