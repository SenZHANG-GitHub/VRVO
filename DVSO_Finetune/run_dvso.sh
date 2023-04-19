# bash run_dev.sh run/val seq exp wStereoPosFlag wCorrectedFlag wGradFlag wStereo scaleEnergyLeftTHR scaleWJI2SumTHR depthoutput warpright
# e.g: bash run_dev.sh run 09 exp Before/After Ori/Corr Hit/Grad 5 2 1 1 1 /home/szha2609


mode=$1
seq=$2
exp=$3
epoch=$4
wStereoPosFlag=$5
wCorrectedFlag=$6
wGradFlag=$7
wStereo=$8
scaleEnergyLeftTHR=$9
scaleWJI2SumTHR=${10}
depthoutput=${11}
warpright=${12}
checkWarpValid=${13}
maskWarpGrad=${14}
home_path=${15}
sub_disp_path=${16}
sub_result_path=${17}


img_path="$home_path/data/kitti/odometry/resized_imgs/$seq"
disp_path="$home_path/src/VRVO/$sub_disp_path"
result_path="$home_path/src/VRVO/$sub_result_path"


model_path="$home_path/src/"
model="DVSO"

# param is now the same as opt.dvso_param
param="$wStereoPosFlag"_"$wCorrectedFlag"_"$wGradFlag"_"$wStereo"_"$scaleEnergyLeftTHR"_"$scaleWJI2SumTHR"_"$warpright"_"$checkWarpValid"_"$maskWarpGrad"


echo "=> Evaluating $exp with virtual stereo residual on KITTI images for seq $seq"

right_disp_name="none"
if [ $warpright = 0 ]
then 
    right_disp_name="disparities_right_pp_640x192.npy"
fi

if [ $warpright = 1 ]
then
    right_disp_name="disparities_right_pp_warped.npy"
fi

echo "=> right_disp_name: $right_disp_name"

if [ $mode = run ] 
then 
    cd $result_path
    rm -rf $param/
    mkdir $param
    cd $param
    
    $model_path/$model/build/bin/dso_dataset \
    files=$img_path/image_2_640x192 \
    calib=$img_path/camera_640x192.txt \
    disps_left=$disp_path/disparities_left_pp_640x192.npy \
    disps_right=$disp_path/$right_disp_name \
    wStereo=$wStereo wStereoPosFlag=$wStereoPosFlag \
    wCorrectedFlag=$wCorrectedFlag wGradFlag=$wGradFlag \
    scaleEnergyLeftTHR=$scaleEnergyLeftTHR scaleWJI2SumTHR=$scaleWJI2SumTHR \
    checkWarpValid=$checkWarpValid maskWarpGrad=$maskWarpGrad \
    judgeHW=1 useVS=1 mode=1 quiet=1 depthoutput=$depthoutput nogui=1

    mkdir depths
    mv *.png depths/

fi

if [ $mode = run ] || [ $mode = val ]
then 
    cd $home_path/src/VRVO/DVSO_Finetune/
    python dev_format.py --seq $seq --result_path $result_path --param $param

    cd $model_path/kitti-odom-eval
    python eval_odom.py --result $result_path/$param/zhan_$seq --seqs $seq --align 7dof

    cd $result_path/$param
    cp zhan_$seq/plot_path/sequence_$seq.pdf .
    cp zhan_$seq/plot_error/*.pdf .
    rm -rf zhan_$seq/plot_path/
    rm -rf zhan_$seq/plot_error/
    rm -rf zhan_$seq/errors/
fi




