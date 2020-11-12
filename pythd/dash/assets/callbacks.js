window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        selectFilter: function(name) {
            let d = document.getElementById('filter-params-div');
            for(let n of d.children) {
                n.style.display = 'none';
            }

            d = document.getElementById(name + '-params-div');
            d.style.display = 'initial';
        },

        compareGroups: function(url) {
            console.log(url);
            if(url !== '') {
                window.open(url, name='_blank');
            }
        }
    }
});

