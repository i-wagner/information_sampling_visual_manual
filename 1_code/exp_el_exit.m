function exp_el_exit(epar)

    if epar.EL

        Eyelink('message', 'Block_End');
        Eyelink('CloseFile');
        Eyelink('ReceiveFile', [], epar.save_path, 1);
        Eyelink('Shutdown');

    end

end