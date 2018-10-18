$(function(){
    $("#register").click(function() {
        $.ajax({
            url : "/register",
            type : "POST",
            dataType:"json",
            contentType : "application/json;charset=UTF-8",
            data : JSON.stringify({
                name: $("#name").val(),	
                id: $("#id").val(),
                password1: $("#password1").val(),
                password2: $("#password2").val(),
                major1: $("#major1").val(),   
            }),             
            success:function(result) {
                alert(result.result);
                if(result.result == "注册成功！"){
                     $('#zhuce').modal('hide');
                }
            },
            error:function(result){
                alert("错误error");
            }
        });
    });

    $("#login").click(function() {
    $.ajax({
        url : "/login",
        type : "POST",
        dataType:"json",
        //contentType : "application/json;charset=UTF-8",
        data : JSON.stringify({
            user: $("#user").val(),	
            pass: $("#pass").val(),

        }),             
        success:function(result) {
            location.href="home";
        },
        error:function(result){
            alert("账号或密码错误");
        }
        });
    });



    $("#renwu").click(function() {
        $.ajax({
            url : "renwu",
            type : "POST",
            dataType:"json",
            contentType : "application/json;charset=UTF-8",
           //向后端传输的数据
            data : JSON.stringify({
            	anpai: $("#jaj").val(),	
            }),             
        	success:function(result) {
        		location.href="home";
            },
            error:function(result){
            //	alert("错误error");
            }
        });
    });

    $("#changePassword").click(function() {
        $.ajax({
            url : "changpassword",
            type : "POST",
            dataType:"json",
            contentType : "application/json;charset=UTF-8",
           //向后端传输的数据
            data : JSON.stringify({
            	   password0: $("#password0").val(),	
            	   password1: $("#password1").val(),	
            	   password2: $("#password2").val(),	
            }),             
            success: function (result) {
            	 alert(result.result);
                 if(result.result == "修改成功！"){
                	 $('#mima').modal('hide');
                 }
        		
            },
            error:function(result){
            	alert("修改失败");
            }
        });
    });
});

