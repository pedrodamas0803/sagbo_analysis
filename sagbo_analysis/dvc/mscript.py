def uncertainty_mesh_size(ref_im: str, def_im: str, roi: tuple, nscale: int = 1):

    assert len(roi) == 6

    xmin, xmax, ymin, ymax, zmin, zmax = roi
    script = [
        f" addpath(genpath('~/UFreckles_PD/'));",
        f"for mesh_size = [8:4:64] ",
        f"    param.analysis='correlation'; ",
        f"    param.reference_image={ref_im};",
        f"    param.deformed_image = {def_im};",
        f"    param.restart=0; ",
        f"    param.result_file=sprintf('unctty_mesh_%d.res', mesh_size); ",
        f"    param.roi=[{xmin}, {xmax}, {ymin}, {ymax}, {zmin}, {zmax}]; ",
        f"    param.pixel_size= 1.000000e+00; ",
        f"    param.convergance_limit=1.000000e-03; ",
        f"    param.iter_max=50; ",
        f"    param.regularization_type='none'; ",
        f"    param.regularization_parameter=1000; ",
        f"    param.psample=1; ",
        f"    model.basis='fem'; ",
        f"    model.nscale={nscale}; ",
        f"    model.mesh_size=mesh_size*[1.000000,1.000000,1.000000]; ",
        f"    nmod=0; ",
        f"    LoadParameters(param); ",
        f"    LoadParameters(model,nmod); ",
        f"    ReferenceImage(nmod); ",
        f"    LoadMeshes(nmod); ",
        f"    LoadMask(nmod); ",
        f"    nscale=model.nscale; ",
        f"",
        f"    U = []; ",
        f"    for iscale=nscale:-1:1   ",
        f"        Uini=InitializeSolution3D(U,nmod,iscale); ",
        f"        ",
        f"    %     if iscale==nscale ",
        f"    %         Uini=0*Uini; ",
        f"    %     end ",
        f"        [U]=Solve3D(Uini,nmod,iscale); ",
        f"    end ",
        f"    unix(['cp ' ,fullfile('TMP','0_error_0.mat'),' ', strrep(param.result_file,'.res','-error.res')]); ",
        f"    load(fullfile('TMP','0_3d_mesh_0'),'Nnodes','Nelems','xo','yo','zo','conn','elt','ng','rint','Smesh','ns'); ",
        f"    save(param.result_file,'U','Nnodes','Nelems','xo','yo','zo','param','model','nmod','conn','elt','ng','rint','Smesh','ns', 'Uini'); ",
        f"    postproVTK3D(param.result_file,0,1); ",
        f"    postproc(param.result_file) ",
        f"end ",
    ]

    return script
