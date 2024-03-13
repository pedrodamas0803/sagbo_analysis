def uncertainty_mesh_size(ref_im:str, def_im:str, roi:tuple, nscale:int = 1):

    assert len(roi) == 6
    
    xmin, xmax, ymin, ymax, zmin, zmax = roi

    script = f" \n
            addpath(genpath('~/UFreckles_PD/'));  \n
            for mesh_size = [8:4:64] \n
                param.analysis='correlation'; \n
                param.reference_image={ref_im}; \n
                param.deformed_image = {def_im}; \n
                param.restart=0; \n
                param.result_file=sprintf('unctty_mesh_%d.res', mesh_size); \n
                param.roi=[{xmin}, {xmax}, {ymin}, {ymax}, {zmin}, {zmax}]; \n
                param.pixel_size= 1.000000e+00; \n
                param.convergance_limit=1.000000e-03; \n
                param.iter_max=50; \n
                param.regularization_type='none'; \n
                param.regularization_parameter=1000; \n
                param.psample=1; \n
                model.basis='fem'; \n
                model.nscale={nscale}; \n
                model.mesh_size=mesh_size*[1.000000,1.000000,1.000000]; \n
                nmod=0; \n
                LoadParameters(param); \n
                LoadParameters(model,nmod); \n
                ReferenceImage(nmod); \n
                LoadMeshes(nmod); \n
                LoadMask(nmod); \n
                nscale=model.nscale; \n
            
                U = []; \n
                for iscale=nscale:-1:1   \n
                    Uini=InitializeSolution3D(U,nmod,iscale); \n
                    
                %     if iscale==nscale \n
                %         Uini=0*Uini; \n
                %     end \n
                    [U]=Solve3D(Uini,nmod,iscale); \n
                end \n
                unix(['cp ' ,fullfile('TMP','0_error_0.mat'),' ', strrep(param.result_file,'.res','-error.res')]); \n
                load(fullfile('TMP','0_3d_mesh_0'),'Nnodes','Nelems','xo','yo','zo','conn','elt','ng','rint','Smesh','ns'); \n
                save(param.result_file,'U','Nnodes','Nelems','xo','yo','zo','param','model','nmod','conn','elt','ng','rint','Smesh','ns', 'Uini'); \n
                postproVTK3D(param.result_file,0,1); \n
                postproc(param.result_file) \n
            end \n
            "
    
    return script