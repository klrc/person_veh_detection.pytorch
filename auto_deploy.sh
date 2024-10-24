
# Step 1: 压缩当前项目
# package_name=$(git branch --show-current)
package_name=person_veh_detection.pytorch
zip -r $package_name.zip . -x "*.git*"

# Step 2: 使用 FTP 自动化删除旧文件并上传
ftp -n -v <<END_SCRIPT
open up
user han.sun Elpsycongroo0
binary
delete $package_name.zip 
put $package_name.zip  
bye
END_SCRIPT

# Step 3: 清除压缩包
rm $package_name.zip