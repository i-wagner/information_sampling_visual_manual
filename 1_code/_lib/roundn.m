function [ out ] = roundn( raw, decimalPlaces )
%ROUNDN Rounds a floating point number to the specified accuracy
out = round(raw * (10^decimalPlaces)) / (10^decimalPlaces);