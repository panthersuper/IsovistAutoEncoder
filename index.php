
<?php
    header('Access-Control-Allow-Origin: *');  


    /*
    Error reporting helps you understand what's wrong with your code, remove in production.
    */

    $input = $_POST["mydata"];
    
    error_reporting(E_ALL); 
    ini_set('display_errors', 1);
    $input = "'".$input."'";

    $term = "/home/design_heritage/platform/dh_upload_ACL/lib/miniconda3/bin/python3 query.py ".$input." 2>&1";
    
    // echo $term;
    $output = shell_exec($term);
    echo $output;











?>
