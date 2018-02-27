
<?php
    header('Access-Control-Allow-Origin: *');  


    /*
    Error reporting helps you understand what's wrong with your code, remove in production.
    */

    $json = $_POST["mydata"];
    // $data = $json->data; //project id to bookmark
    
    error_reporting(E_ALL); 
    ini_set('display_errors', 1);

    $output = shell_exec("/home/design_heritage/platform/dh_upload_ACL/lib/miniconda3/bin/python3 query.py ".$json." 2>&1");

    echo $output;
    // echo $_POST["mydata"];











?>
