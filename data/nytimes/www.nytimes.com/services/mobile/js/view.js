$(document).ready(function()
{
	
	var str= window.location.href;
	var aview = "";
	var isbrowser = "";
	if (str.indexOf('?') >= 0 && str.substring(str.indexOf('?'),str.indexOf('&')).indexOf('id=') !== -1) {
		aview = "product";
		getUrlVars();
		var productid = getUrlVars()["id"];
		var model = getUrlVars()["model"];
		var device = getUrlVars()["device"];
		$('.container').addClass("prod-page");
		jview(model,productid);
		jfilter(device,model,productid);
	}	
	else
	{
		aview="index";
		$('#main h2').html("Introducing The New York Times Cooking app");
		$('#main h4').html("The essential ingredient, for iPad<sup class='sm'>&reg;</sup>, iPhone<sup class='sm'>&reg;</sup> and Apple Watch<sup class='sm'>&reg;</sup>");
		$('#main-btn').html('<a class="button-badge appstore" href="https://itunes.apple.com/us/app/nyt-cooking-recipes-from-new/id911422904?mt=8" target="_blank" style="opacity: 1;">Available on the App Store</a>');
		$('#device-img').attr("src", "img/NYT_Cooking-iPad_AW1.png");
		jfilter("all","all","zero");
	}


	$(".nav").click(function(e){
		e.preventDefault();
		$(this).parent().addClass("active");
		$(this).parent().siblings().removeClass("active");
		var str = (e.target.id).substring(4,(e.target.id).length);   
		
		if($.browser.msie && parseFloat($.browser.version) < 8){
	    	isbrowser = "ie7";
		}
		
		if (str == "dev-all")
		{
			if (isbrowser == "ie7"){
			$('#appscontainer article').show();
			}
			else{
				$('#appscontainer article').fadeOut('fast');
				$('#appscontainer article').promise().done(function() {
					$('#appscontainer article').fadeIn('fast');
				});
			}
		}
		
		else if (str == "dev-phone-mod-all")
		{
			if (isbrowser == "ie7"){
			$('#appscontainer article').hide();
			$('.dev-phone').show();
			}
			else{
				$('#appscontainer article').fadeOut('fast');
				$('#appscontainer article').promise().done(function() {
					$('.dev-phone').fadeIn('fast');
				});
			}
		}
		
		else if (str == "dev-tablet-mod-all")
		{
			if (isbrowser == "ie7"){
				$('#appscontainer article').hide();
			    $('.dev-tablet').show();
				}
			else{
				$('#appscontainer article').fadeOut('fast');
				$('#appscontainer article').promise().done(function() {
					$('.dev-tablet').fadeIn('fast');
				});
			}
		}
		
		else
		{
			if (isbrowser == "ie7"){
			$('#appscontainer article').hide();
			$('.'+str).show();
			}
			else{
				$('#appscontainer article').fadeOut('fast');
				$('#appscontainer article').promise().done(function() {
					$('.'+str).fadeIn('fast');
				});
			}
		}
		
		return false;
	});


}); // end $(document).ready()






function getUrlVars()
{
    var vars = [], hash;
    var hashes = window.location.href.slice(window.location.href.indexOf('?') + 1).split('&');
    for(var i = 0; i < hashes.length; i++)
    {
        hash = hashes[i].split('=');
        vars.push(hash[0]);
        vars[hash[0]] = hash[1];
    }
    return vars;
}


function jview (model,id)
{	
	
	var flength = '';
	var tdata = '';
	var appDbUrl = "js/app-db-product.json"

	$.ajax
	({type: "GET",url: appDbUrl,dataType:"json",success: function(response)
	{
		var flength = '';
		flength = response.length; 

		for(i=0;i<flength;i++)
		{
		data =response[i];


		if (data.id==id)
		{	

			var acopy=data.copy;
			var headline=data.headline;
			var alinktext=data.alinktext;
			var alink=data.alink;
			var headlinea=data.topheadlinea;
			var headlineb=data.topheadlineb;
			var astore=data.astore;
			var storelink=data.storelink;
			var imgpath=data.imgpath;
			tdata="<h2>"+headline+"</h2><p>"+acopy+"</p><p><a href='"+alink+"' target='_blank'>"+alinktext+"</a></p>";

			if (data.id == 5){
				var disclaimer = data.disclaimer;
				tdata += "<p class='disclaimer'>"+disclaimer+"</p>";
			}

			var btnclass='app-box-button';

			if (model=='ANDROID')
			{
				btnclass='button-badge googleplay';
			}
			else if (model=='iPHONE'||model=='iPAD')
			{
				btnclass='button-badge appstore';
			}
			else if (model=='KINDLE')
			{
				btnclass='button-badge amazonapps';
			}
			else if (model=='WINDOWS%208') 
			{
				btnclass='button-badge windows'; 
			}

			btndata="<a class='"+btnclass+"' href='"+storelink+"' target='_blank'>"+astore+"</a>";
		}

		}

		$('#proddetails .maincontent').html(tdata);
		$('#main h2').html(headlinea);
		$('#main h4').html(headlineb);
		$('#main-btn').html(btndata);
		$('#device-img').attr("src", imgpath);

	}

	});

}



function jfilter (jdevice,jmodel,jid)
{	
	
	var flength = '';
	var tdata = '';
	var appDbUrl = "js/app-db.json"
	var jarr = new Array();
	var jarri = 0;
	var jhits=0;


	$.ajax
	({type: "GET",url: appDbUrl,dataType:"json",success: function(response)
	{
		var flength = '';
		flength = response.length; 

		if (jid=="zero"){
		    response = response.sort(function(a, b) {
		       return (a.rank-b.rank);
		    });	
		} else {
			/* update meta tags */
			$('meta[name="DCS.dcuri"]').attr('content', window.location.href);
			//Dec 8: the below line seems to be intended to adjust the page title .. but it throws an error .. for Cooking.. So, comment out until find why... 
			//document.title = document.title+" - "+response[jid-1].appsrc.replace(/<(?:.|\n)*?>/gm, '').replace(/&reg;|&trade;/g, '').replace(/&amp;/, '&').replace(/&nbsp;/, ' ');
			$('head #WTti').attr('content', document.title);	
		}

		for(i=0;i<flength;i++)
		{
		data =response[i];
		var rank=data.rank;
		var amodel=data.model;
		var appsrc=data.appsrc;
		var image=data.image;
		var desc=data.desc;
		var aid=data.id;
		var alink=data.alink;
		var aclass=data.aclass;
		var store=data.store;
		var tlist="";
		var adevice=data.device;
		var containsJdevice = adevice.indexOf(jdevice) >= 0;

		for(j=0;j<data.alist.length; j++)
		{
			tlist+="<li>"+data.alist[j].txt+"</li>";
		}

		var jstr = "<article id='"+aid+"' class='pure-u-1-3 "+aclass+"'><div class='app-box'><a class='app-box-full-link m' href='index.html?id="+aid+"&device="+adevice+"&model="+amodel+"'></a><div class='app-box-header'><img class='app-box-img' src='"+image+"'/><div class='app-box-header-content'><div class='model'>"+amodel+"</div><div class='appsrc'>"+appsrc+"</div></div><a class='app-box-button m' href='index.html?id="+aid+"&device="+adevice+"&model="+amodel+"'>Learn More &raquo;</a></div><div class='app-box-main'>"+desc+"<ul>"+tlist
    
		if(aid==18 || aid==19) {  // IF VR for Google, iOS make Learn MOre go to the VR page 
		 jstr = "<article id='"+aid+"' class='pure-u-1-3 "+aclass+"'><div class='app-box'><a class='app-box-full-link m' href='http://www.nytimes.com/marketing/nytvr/'></a><div class='app-box-header'><img class='app-box-img' src='"+image+"'/><div class='app-box-header-content'><div class='model'>"+amodel+"</div><div class='appsrc'>"+appsrc+"</div></div><a class='app-box-button m' href='index.html?id="+aid+"&device="+adevice+"&model="+amodel+"'>Learn More &raquo;</a></div><div class='app-box-main'>"+desc+"<ul>"+tlist
	
		jstr+="<li class='learnmore'><a href='http://www.nytimes.com/marketing/nytvr/'>Learn More &raquo;</a></li></ul></div><a class='app-box-button' href='"+alink+"' target='_blank'>"+store+"</a></div></article>";

		} else {

		jstr+="<li class='learnmore'><a href='index.html?id="+aid+"&device="+adevice+"&model="+amodel+"'>Learn More &raquo;</a></li></ul></div><a class='app-box-button' href='"+alink+"' target='_blank'>"+store+"</a></div></article>";
        }

		if((containsJdevice && (jmodel==amodel))  || (containsJdevice && (jmodel=="all")) || (jdevice=="all"))
		{
			
		if ((jid!="zero")&&(jid!=aid)&&(jhits<4)){
			tdata += jstr;
			jhits++;
		}
		else if (jid=="zero"){
			tdata += jstr;
		}

		}

		else if (jid!=aid)
		{
			jarr[jarri]=jstr;
			jarri++;
		}

		}

		if((jhits<4)&&(jid!="zero"))
		{
			for(j=jhits;j<4;j++)
			{
				tdata+=jarr[j];	
			}
		}

		tdata+="<div></div>";

		$('#appscontainer .sectionwrapper').html(tdata);
		
		$('.app-box-button, .button-badge').hover(
   			function() {
         	$(this).fadeTo('fast', 0.8);
  			},
   			function() {
         	$(this).fadeTo('fast', 1);
  			}
		);
        
        $("#1 .learnmore").remove();

	}
    

	});
	


}
