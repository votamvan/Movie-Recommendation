module.exports = class Helper {
    getUserInfo() {
        const defaults = { userId: 0, displayName: '', userToken: '', email: '', messageBox: {} };
        try {
            const stored = JSON.parse(localStorage.getItem('userInfo'));
            return Object.assign(defaults, stored && typeof stored === 'object' ? stored : {});
        } catch (error) {
            localStorage.removeItem('userInfo');
            return defaults;
        }
    }
    setUserInfo(data) {
        localStorage.setItem('userInfo', JSON.stringify(data));
    }
    unsetUserInfo() {
        localStorage.removeItem('userInfo');
    }
    setLocalMessage(status, message) {
        const user = this.getUserInfo();
        user.messageBox = { status, message };
        this.setUserInfo(user);
    }
    getLocalMessage() {
        const user = this.getUserInfo();
        const message = user.messageBox || {};
        user.messageBox = {};
        this.setUserInfo(user);
        message.alert_status = message.status === 'success' ? 'alert alert-success' : 'alert alert-danger';
        return message;
    }
    showMessage(status, message) {
        window.alert(message);
    }
    async request(params) {
        const loader = document.querySelector('#loader');
        this.pending = (this.pending || 0) + 1;
        if (loader) loader.style.display = 'block';
        try {
            const headers = { 'Content-Type': 'application/json' };
            const token = this.getUserInfo().userToken;
            if (params.auth !== false && token) headers.Authorization = 'Bearer ' + token;
            const response = await fetch(params.url, {
                method: params.type || 'POST', headers,
                body: params.data === undefined ? undefined : JSON.stringify(params.data)
            });
            const data = await response.json();
            if (response.status === 401 && params.auth !== false) {
                this.unsetUserInfo();
                window.location.href = '/login';
                return;
            }
            if (!response.ok || data.status === 'error') {
                if (params.error) params.error(data);
                else this.showMessage('danger', data.message || 'Request failed.');
                return;
            }
            if (params.success) params.success(data);
        } catch (error) {
            const data = { status: 'error', message: 'Cannot reach the server. Please try again.' };
            if (params.error) params.error(data);
            else this.showMessage('danger', data.message);
        } finally {
            this.pending--;
            if (loader && this.pending === 0) loader.style.display = 'none';
            if (params.complete) params.complete();
        }
    }
}
