def uncertainty_mesh_size(ref_im:str, def_im:str, roi:tuple, nscale:int = 1):

    assert len(roi) == 6
    
    xmin, xmax, ymin, ymax, zmin, zmax = roi

    script = f" addpath(genpath('~/UFreckles_PD/'));  \\
            for mesh_size = [8:4:64] 
                param.analysis='correlation'; 
                param.reference_image={ref_im};
                param.deformed_image = {def_im};
                param.restart=0; 
                param.result_file=sprintf('unctty_mesh_%d.res', mesh_size); 
                param.roi=[{xmin}, {xmax}, {ymin}, {ymax}, {zmin}, {zmax}]; 
                param.pixel_size= 1.000000e+00; 
                param.convergance_limit=1.000000e-03; 
                param.iter_max=50; 
                param.regularization_type='none'; 
                param.regularization_parameter=1000; 
                param.psample=1; 
                model.basis='fem'; 
                model.nscale={nscale}; 
                model.mesh_size=mesh_size*[1.000000,1.000000,1.000000]; 
                nmod=0; 
                LoadParameters(param); 
                LoadParameters(model,nmod); 
                ReferenceImage(nmod); 
                LoadMeshes(nmod); 
                LoadMask(nmod); 
                nscale=model.nscale; 
            
                U = []; 
                for iscale=nscale:-1:1   
                    Uini=InitializeSolution3D(U,nmod,iscale); 
                    
                %     if iscale==nscale 
                %         Uini=0*Uini; 
                %     end 
                    [U]=Solve3D(Uini,nmod,iscale); 
                end \
                unix(['cp ' ,fullfile('TMP','0_error_0.mat'),' ', strrep(param.result_file,'.res','-error.res')]); 
                load(fullfile('TMP','0_3d_mesh_0'),'Nnodes','Nelems','xo','yo','zo','conn','elt','ng','rint','Smesh','ns'); 
                save(param.result_file,'U','Nnodes','Nelems','xo','yo','zo','param','model','nmod','conn','elt','ng','rint','Smesh','ns', 'Uini'); 
                postproVTK3D(param.result_file,0,1); 
                postproc(param.result_file) 
            end 
            "
    
    return script