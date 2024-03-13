def uncertainty_mesh_size(ref_im: str, def_im: str, roi: tuple, nscale: int = 1):

    assert len(roi) == 6

    xmin, xmax, ymin, ymax, zmin, zmax = roi
    script = [
        f" addpath(genpath('~/UFreckles_PD/'));\n",
        f"for mesh_size = [8:4:64] \n",
        f"    param.analysis='correlation'; \n",
        f"    param.reference_image='{ref_im}';\n",
        f"    param.deformed_image = '{def_im}';\n",
        f"    param.restart=0; \n",
        f"    param.result_file=sprintf('unctty_mesh_%d.res', mesh_size); \n",
        f"    param.roi=[{xmin}, {xmax}, {ymin}, {ymax}, {zmin}, {zmax}]; \n",
        f"    param.pixel_size= 1.000000e+00; \n",
        f"    param.convergance_limit=1.000000e-03; \n",
        f"    param.iter_max=50; \n",
        f"    param.regularization_type='none'; \n",
        f"    param.regularization_parameter=1000; \n",
        f"    param.psample=1; \n",
        f"    model.basis='fem'; \n",
        f"    model.nscale={nscale}; \n",
        f"    model.mesh_size=mesh_size*[1.000000,1.000000,1.000000]; \n",
        f"    nmod=0; \n",
        f"    LoadParameters(param); \n",
        f"    LoadParameters(model,nmod); \n",
        f"    ReferenceImage(nmod); \n",
        f"    LoadMeshes(nmod); \n",
        f"    LoadMask(nmod); \n",
        f"    nscale=model.nscale; \n",
        f"\n",
        f"    U = []; \n",
        f"    for iscale=nscale:-1:1   \n",
        f"        Uini=InitializeSolution3D(U,nmod,iscale); \n",
        f"        \n",
        f"    %     if iscale==nscale \n",
        f"    %         Uini=0*Uini; \n",
        f"    %     end \n",
        f"        [U]=Solve3D(Uini,nmod,iscale); \n",
        f"    end \n",
        f"    unix(['cp ' ,fullfile('TMP','0_error_0.mat'),' ', strrep(param.result_file,'.res','-error.res')]); \n",
        f"    load(fullfile('TMP','0_3d_mesh_0'),'Nnodes','Nelems','xo','yo','zo','conn','elt','ng','rint','Smesh','ns'); \n",
        f"    save(param.result_file,'U','Nnodes','Nelems','xo','yo','zo','param','model','nmod','conn','elt','ng','rint','Smesh','ns', 'Uini'); \n",
        f"    postproVTK3D(param.result_file,0,1); \n",
        f"    postproc(param.result_file) \n",
        f"end \n",
    ]

    return script


def uncertainty_lambda_size(
    ref_im: str, def_im: str, mesh_size: int, roi: tuple, nscale: int = 1
):

    assert len(roi) == 6

    xmin, xmax, ymin, ymax, zmin, zmax = roi
    script = [
        f" addpath(genpath('~/UFreckles_PD/'));\n",
        f"mesh_size = {mesh_size}\n",
        f"for lambda = [0.5:0.1:2]*mesh_size \n",
        f"    param.analysis='correlation'; \n",
        f"    param.reference_image='{ref_im}';\n",
        f"    param.deformed_image = '{def_im}';\n",
        f"    param.restart=0; \n",
        f"    param.result_file=sprintf('unctty_lambda_%d.res', lambda); \n",
        f"    param.roi=[{xmin}, {xmax}, {ymin}, {ymax}, {zmin}, {zmax}]; \n",
        f"    param.pixel_size= 1.000000e+00; \n",
        f"    param.convergance_limit=1.000000e-03; \n",
        f"    param.iter_max=50; \n",
        f"    param.regularization_type='tiko'; \n",
        f"    param.regularization_parameter=lambda; \n",
        f"    param.psample=1; \n",
        f"    model.basis='fem'; \n",
        f"    model.nscale={nscale}; \n",
        f"    model.mesh_size=mesh_size*[1.000000,1.000000,1.000000]; \n",
        f"    nmod=0; \n",
        f"    LoadParameters(param); \n",
        f"    LoadParameters(model,nmod); \n",
        f"    ReferenceImage(nmod); \n",
        f"    LoadMeshes(nmod); \n",
        f"    LoadMask(nmod); \n",
        f"    nscale=model.nscale; \n",
        f"\n",
        f"    U = []; \n",
        f"    for iscale=nscale:-1:1   \n",
        f"        Uini=InitializeSolution3D(U,nmod,iscale); \n",
        f"        \n",
        f"    %     if iscale==nscale \n",
        f"    %         Uini=0*Uini; \n",
        f"    %     end \n",
        f"        [U]=Solve3D(Uini,nmod,iscale); \n",
        f"    end \n",
        f"    unix(['cp ' ,fullfile('TMP','0_error_0.mat'),' ', strrep(param.result_file,'.res','-error.res')]); \n",
        f"    load(fullfile('TMP','0_3d_mesh_0'),'Nnodes','Nelems','xo','yo','zo','conn','elt','ng','rint','Smesh','ns'); \n",
        f"    save(param.result_file,'U','Nnodes','Nelems','xo','yo','zo','param','model','nmod','conn','elt','ng','rint','Smesh','ns', 'Uini'); \n",
        f"    postproVTK3D(param.result_file,0,1); \n",
        f"    postproc(param.result_file) \n",
        f"end \n",
    ]

    return script


def slurm_script():
    pass
