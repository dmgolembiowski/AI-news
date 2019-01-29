/* apple-pay.js */
var ap = 0;
var cid = getParameterByName("campaignId");
//Apple Pay email required ..
var emailRequired = false;

function checkForAP() {

	if (getCookie('NYT-S') === '' || getCookie('NYT-S') === null || getCookie('NYT-S').length < 81) {
		emailRequired = true; // if user not logged into NYT then require Apple Pay to set a email
	}
	var paymentInfoCookingMonthly = {
		countryCode: 'US',
		currencyCode: 'USD',
		offer: {
			name: 'Cooking Monthly',
			price: 5.00,
			type: 'regular-digital',
			identification: {
				offerChainId: 20000124820,
				campaignId: cid
			}
		},
		edu: false,
		totalLabel: 'NYTimes',
		requireEmail: emailRequired
	};
	var paymentInfoCookingAnnual = {
		countryCode: 'US',
		currencyCode: 'USD',
		offer: {
			name: 'Cooking Annual',
			price: 40,
			type: 'regular-digital',
			identification: {
				offerChainId: 20000140020,
				campaignId: cid
			}
		},
		edu: false,
		totalLabel: 'NYTimes',
		requireEmail: emailRequired
	};
	//if (apple = 1) {
	if (window.ApplePaySession && window.ApplePaySession.canMakePayments()) {
		ap = 1;
		var serviceHostName = 'https://myaccount.nytimes.com';
		if (getParameterByName("svcHost") === "stg") {
			serviceHostName = 'https://myaccount-circ.stg.nytimes.com'; //
			// console.log("use prod svc host");
		}
		// Append Apple pay script
		var script = document.createElement('script');
		script.src = serviceHostName + '/get-started/js/dest/apple-pay.js?t=1';
		document.head.appendChild(script);
		// Once script is loaded
		script.onload = function() {
			window.NYTAPPY.setServiceHostName(serviceHostName);
			//get the Apple Pay content
			var ap_content_url = "../components/bundledetails/express_checkout_applepay_vary2.html";
			$.get(ap_content_url, function(applePayContent) {

				var monthlyLegal = '<strong>Express Checkout</strong> Your payment method will be automatically charged $5 every 4 weeks in advance. Your subscription automatically renews and your payment method is charged in advance of each billing period unless you cancel. You may <a href="http://www.nytimes.com/content/help/rights/sale/terms-of-sale.html#cancel">cancel</a> at any time. By subscribing, you are also accepting the <a href="http://www.nytimes.com/content/help/rights/terms/terms-of-service.html">Terms of Service</a>, <a href="https://www.nytimes.com/privacy">Privacy Policy</a>, and <a href="http://www.nytimes.com/content/help/rights/sale/terms-of-sale.html">Terms of Sale</a>, including the <a href="http://www.nytimes.com/content/help/rights/sale/terms-of-sale.html#cancel">Cancellation Policy.</a>';

				var annuallyLegal = '<strong>Express Checkout</strong> Your payment method will be automatically charged $40 for a one-year subscription. Your subscription automatically renews annually, and your payment method is charged in advance unless you cancel. You may <a href="http://www.nytimes.com/content/help/rights/sale/terms-of-sale.html#cancel">cancel</a> at any time. By subscribing, you are also accepting the <a href="http://www.nytimes.com/content/help/rights/terms/terms-of-service.html">Terms of Service</a>, <a href="https://www.nytimes.com/privacy">Privacy Policy</a>, and <a href="http://www.nytimes.com/content/help/rights/sale/terms-of-sale.html">Terms of Sale</a>, including the <a href="http://www.nytimes.com/content/help/rights/sale/terms-of-sale.html#cancel">Cancellation Policy.</a>';

				// Remove click handlers from larger bundle containers
				$('.bundle_prod').off();
				// console.log('remove bundle click');
				// Append Apple Pay content to Basic bundle
				$('#cooking-monthly .ap_below').after(applePayContent);
				// Append Apple Pay content to Basic bundle
				$('#cooking-annual .ap_below').after(applePayContent);
				// Append Apple Pay content to HD bundle
				// Set to same content to achieve equal height, even though content is hidden
				
				
				
				// Removes padding from Apple Pay info
				$('.apple_pay_holder').css('paddingBottom', '0');
                $('.apple_pay_holder .applePayBtn').removeClass("apple-pay-button-white-with-text");
                $('.apple_pay_holder .applePayBtn').addClass("apple-pay-button-black-with-text");
				// Replaces HD bundle with dummy content
				//change the regular CTA text to say Credit Card
				$('a.cta').html('Pay with Credit Card');
				//show the AP content
				$('body').addClass("applePay vary");
				$(".billed_price").show();
				// $("#bundle_hd .billed_price").html("<br/>");
				$('.apple_pay_holder p a').attr("target", "_blank"); //links to Cancellation policy etc
				$('.apple_pay_holder').show();
				$(".bundle_tab").off(); //remove the full bundle desktop behavior that goes to checkout


				// Append Apple Pay Legal for Monthly
				$('#cooking-monthly .ap-legal').html(monthlyLegal);
				// Append Apple Pay Legal for Anually
				$('#cooking-annual .ap-legal').html(annuallyLegal);

				//
				/// this offer code is for product that uses crosswords, for Feb sale  ... //
				var bundleClicked = 'monthly';
				//activate the buttons
				var errorCallback = function(error) {
					//where to put the error message
					console.log("bundle clicked is " + bundleClicked);
					if (bundleClicked === "annual") {
						$("#bundle_all_access .ap_errors").html(error.message);
					} else {
						$("#bundle_basic .ap_errors").html(error.message);
					}
					document.dispatchEvent(new window.CustomEvent('appy-error', {
						detail: { 'error-code': error.code }
					}));
				};
				// Add click handler to Basic bundle Apple Pay button
				$('#cooking-monthly .applePayBtn').on("click", function() {
					window.NYTAPPY.startPaymentSession(paymentInfoCookingMonthly, errorCallback);
					bundleClicked = "monthly";
				});
				// Add click handler to Better bundle Apple Pay button
				$('#cooking-annual .applePayBtn').on("click", function() {
					window.NYTAPPY.startPaymentSession(paymentInfoCookingAnnual, errorCallback);
					bundleClicked = "annual";
				});
			}); //end of loaded content ..
		} //end of script loaded function
	} //end of If AP capable
} //end of Function check for AP
