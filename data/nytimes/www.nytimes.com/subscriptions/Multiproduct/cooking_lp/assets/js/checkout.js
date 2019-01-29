// Gets parameters, given their name
function getParameterByName(name) {
	name = name.replace(/[\[]/, "\\\[").replace(/[\]]/, "\\\]");
	var regexS = "[\\?&]" + name + "=([^&#]*)";
	var regex = new RegExp(regexS);
	var results = regex.exec(window.location.search);
	if (results == null)
		return "";
	else
		return decodeURIComponent(results[1].replace(/\+/g, " "));
}

// modify link URLs with campaign ID and offer chain
function addLinkParameters(linkClass) {

	//alert("link param function");
	// Check for campaign ID
	if (window.location.href.indexOf("campaignId=") > -1) {
		// Grab campaign ID
		var cid = getParameterByName('campaignId');
		// Construct partial query string for campaign ID
		var campaignIdString = 'campaignId=' + cid;
	}
	// for each purchase link
	$(linkClass).each(function () {
		// debug
		// console.group($(this).data('linkname'));
		// reset vars
		var offerChainString = '';
		var newURL = '';
		// grab current url
		var existingURL = $(this).attr("href");
		// grab offer chain number
		var OC = $(this).data('oc');
		// Construct partial query string for OC
		var offerChainString = 'OC=' + OC;
		// build new URL
		newURL = existingURL + '?' + offerChainString;
		// Check for campaign ID
		if (campaignIdString !== undefined) {
			newURL += '&' + campaignIdString;
		}
		// Set anchor tag's href to new URL
		$(this).attr("href", newURL);
		// debug
		// console.log('Campaign ID:', cid);
		// console.log('Campaign ID String:', campaignIdString);
		// console.log('Offer Chain:', OC);
		// console.log('Offer Chain String:', offerChainString);
		// console.log('New URL:', newURL);
		// console.groupEnd();

	});
}

(function ($) {
	$(document).ready(function () {
		//var params = unescape(window.location.search.replace('?', '&').replace(/OC=\d+&/, ''));
		//var params2 = params.replace("campaignId", "CMP");
		
	});
})(jQuery);
