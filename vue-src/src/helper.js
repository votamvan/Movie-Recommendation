module.exports = class Helper {

    userRole = {
        user:'AUDIENCE'
    }   
    
    getUserInfo () {
            var userInfo = {
                userId: 0,
                displayName: '',
                userToken: '',
                email: '',
                role: '',
                profilePic: '',
                messageBox:''
            };
            var userInfoLocal = JSON.parse(localStorage.getItem('userInfo'));
            if (typeof (userInfoLocal) === 'object') {
                if (userInfoLocal !== null) {
                    if (typeof (userInfo.userId) !== 'undefined') {
                        if (userInfoLocal.userId !== 0) {
                            userInfo.userId = userInfoLocal.userId;
                        }
                    }
                    if (typeof (userInfo.displayName) !== 'undefined') {
                        if (userInfoLocal.displayName !== '') {
                            userInfo.displayName = userInfoLocal.displayName;
                        }
                    }
                    if (typeof (userInfo.userToken) !== 'undefined') {
                        if (userInfoLocal.userToken !== '') {
                            userInfo.userToken = userInfoLocal.userToken;
                        }
                    }
                    if (typeof (userInfo.email) !== 'undefined') {
                        if (userInfoLocal.email !== '') {
                            userInfo.email = userInfoLocal.email;
                        }
                    }
                    if (typeof (userInfo.role) !== 'undefined') {
                        if (userInfoLocal.role !== '') {
                            userInfo.role = userInfoLocal.role;
                        }
                    }
                    if (typeof (userInfo.profilePic) !== 'undefined') {
                        if (userInfoLocal.profilePic !== '') {
                            userInfo.profilePic = userInfoLocal.profilePic;
                        }
                    }
                    if (typeof (userInfo.messageBox) !== 'undefined') {
                        if (typeof(userInfoLocal.messageBox.status) != 'undefined') {
                            userInfo.messageBox = userInfoLocal.messageBox;
                            if(typeof(userInfo.messageBox.status) == 'undefined'){
                                userInfo.messageBox['status'] == '';
                                userInfo.messageBox['message'] == '';
                            }
                        }
                    }
                }

            }
            return userInfo;
    }
    setLocalMessage(status,message){
        var userInfo = this.getUserInfo();
        userInfo['messageBox'] = {
            status:status,
            message:message
        };
        this.setUserInfo(userInfo);
    }
    getLocalMessage(){
        var userInfo = this.getUserInfo();
        var messageBox = userInfo.messageBox;
        userInfo['messageBox'] = {};
        this.setUserInfo(userInfo);
        if(typeof(messageBox.status) != 'undefined'){
            if(messageBox.status == 'success'){
                messageBox['alert_status'] = 'alert alert-success';
            }else if(messageBox.status == 'danger'){
                messageBox['alert_status'] = 'alert alert-danger';
            }
        }        
        return messageBox;
    }
    setUserNewToken = function (newToken) {
        var userInfo = this.getUserInfo();
        if (userInfo.identity != '') {
            var userInfoLocal = JSON.parse(localStorage.getItem('userInfo'));
            userInfoLocal.userToken = newToken;
            this.setUserInfo(userInfoLocal);
        }
    }
    unsetUserInfo() {
        localStorage.removeItem('userInfo');
    }
    setUserInfo(datas) {  
        if(typeof(datas.messageBox) == 'undefined'){
            datas['messageBox'] = {}
        }
        localStorage.setItem('userInfo', JSON.stringify(datas));
    }
    getFormData(formData){

        // formData.append('identity',this.getUserInfo().identity);
        // formData.append('displayName',this.getUserInfo().displayName);

        return formData;
    }
    request(params) {

            if (typeof (params) === 'undefined') {
                alert('Empty parameter found.');

            } else {
                if (typeof (params.url) === 'undefined') {
                    alert('Api url is missing');
                } else {
                    
                    var method = 'POST';
                    if (typeof (params.type) !== 'undefined') {
                        method = params.type;
                    }
                    params.type = method;
                    if (typeof(params.success) == 'function') {//start

                        if (window.$) {

                            var ajaxParms = {};
                            ajaxParms['dataType'] = 'json';

                            if(typeof(params.auth) == 'undefined'){
                                ajaxParms['headers'] = {
                                    'Authorization':'Bearer '+this.getUserInfo().userToken
                                };
                            }else{
                                if(params.auth == true){
                                    ajaxParms['headers'] = {
                                        'Authorization':'Bearer '+this.getUserInfo().userToken
                                    };

                                }
                                
                            }
                            ajaxParms['beforeSend'] = ()=>{
                                document.querySelector('#loader').style.display = 'block';
                            };
                            ajaxParms['url'] = params.url;
                            ajaxParms['type'] = params.type;
                            ajaxParms['xhr'] = function () {
                                                    var myXhr = window.$.ajaxSettings.xhr();
                                                    if (typeof(params.uploadProgress) == 'function') {
                                                        if (myXhr.upload) {
                                                            myXhr.upload.addEventListener('progress', function (e) {
                                                                var uploaded = e.loaded;
                                                                var total_size = e.total;
                                                                var percentage = (uploaded / total_size) * 100;
                                                                if (percentage > 100) {
                                                                    percentage = 100;
                                                                }
                                                                var return_datas = {
                                                                    uploaded: uploaded,
                                                                    total_size: total_size,
                                                                    uploading_percent: percentage,
                                                                };
                                                                params.uploadProgress(return_datas);
                                                            }, false);
                                                        }
                                                    }
                                                    return myXhr;
                                                };
                            ajaxParms['complete'] = (resp) =>{
                                    document.querySelector('#loader').style.display = 'none';
                                    
                                    if (typeof(params.complete) == 'function') {
                                        if(resp.status == 403){//credials
                                            params.complete(resp);
                                        }else if(resp.status == 404){
                                            this.showMessage('danger','Record not found.');
                                        }else if(resp.status == 400){
                                            this.showMessage('danger','Bad Request found.');
                                        }else if(resp.status == 500){
                                            this.showMessage('danger','Internal Server Error.');
                                        }
                                        
                                    }else{
                                        if(resp.status == 403){//credials
                                            this.showMessage('danger','Username or password is not matched.');
                                        }else if(resp.status == 404){
                                            this.showMessage('danger','Record not found.');
                                        }else if(resp.status == 400){
                                            this.showMessage('danger','Bad Request found.');
                                        }
                                    }
                                    
                                };
                            ajaxParms['success'] = (resp) =>{

                                    if (typeof(params.success) == 'function') {
                                        params.success(resp);
                                    }
                            };
                            if (typeof(params.beforeSend) == 'function') {
                                params.beforeSend();
                            }
                            if(typeof(params.withData) != 'undefined'){
                                if(params.withData == 'json'){

                                    ajaxParms['data'] = JSON.stringify(params.data);
                                    ajaxParms['contentType'] = 'application/json';
                                    
                                    window.$.ajax(ajaxParms);
                                }else{

                                    ajaxParms['data'] = params.data;
                                    ajaxParms['cache'] = false;
                                    ajaxParms['contentType'] = false;
                                    ajaxParms['processData'] = false;
                                    window.$.ajax(ajaxParms);
                                }
                            }else{//default is form data

                                ajaxParms['data'] = params.data;
                                ajaxParms['cache'] = false;
                                ajaxParms['contentType'] = false;
                                ajaxParms['processData'] = false;
                                window.$.ajax(ajaxParms);
                            }

                            
                        } else {
                            console.log('jquery not loaded');
                        }

                    }//end      

                }
            }
    }

    scroll_to_bottom(selector) {
        var objDiv = document.querySelector(selector);
        if (objDiv) {
            objDiv.scrollTop = objDiv.scrollHeight;
        }
    }

}
