clear all; close all; clc;
[sumChoice sumFixChoice sumFixSet] = decodeRecursiveProb(1,[2 8]);
fprintf('Fixations (set1, set2, both):\t\t\t%.2f,\t%.2f,\t%.2f\n',sumFixSet(1),sumFixSet(2),sum(sumFixSet));
fprintf('Fixations (chosen, nonchosen, both):\t%.2f,\t%.2f,\t%.2f\n',sumFixChoice(1),sumFixChoice(2),sum(sumFixChoice));
fprintf('Targets (set1, set2, both):\t\t\t\t%.2f,\t%.2f,\t%.2f\n',sumChoice(1),sumChoice(2),sum(sumChoice));