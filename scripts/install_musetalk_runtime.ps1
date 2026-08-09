$ErrorActionPreference = "Stop"

& python -m pip install `
    "mmengine==0.10.7" `
    "mmcv-lite==2.0.1" `
    "mmdet==3.1.0" `
    "pycocotools==2.0.11" `
    "json-tricks==3.17.3" `
    "munkres==1.1.4"
if ($LASTEXITCODE -ne 0) {
    throw "OpenMMLab runtime installation failed"
}

# MMPose declares optional dataset packages that do not ship Python 3.12
# Windows wheels. The local xtcocotools compatibility package covers the
# COCO imports used while initializing MuseTalk's RTMPose model.
& python -m pip install --no-deps "mmpose==1.1.0"
if ($LASTEXITCODE -ne 0) {
    throw "MMPose installation failed"
}

Write-Host "MuseTalk OpenMMLab runtime installed."
